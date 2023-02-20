import tensorflow as tf
from tensorflow import keras
from einops import rearrange, repeat
from opt_einsum import contract, contract_expression
import math
import numpy as np
from scipy import special as ss


def power(L, A, v=None):
    """ Compute A^L and the scan sum_i A^i v_i
    A: (..., N, N)
    v: (..., N, L)
    """
    I = tf.eye(A.shape[-1], dtype=A.dtype) # , dtype=A.dtype, device=A.device)

    powers = [A]
    l = 1
    while True:
        if L % 2 == 1: I = powers[-1] @ I
        L //= 2
        if L == 0: break
        l *= 2
        powers.append(powers[-1] @ powers[-1])

    if v is None: return I

    # Invariants:
    # powers[-1] := A^l
    # l := largest po2 at most L

    # Note that an alternative divide and conquer to compute the reduction is possible and can be embedded into the above loop without caching intermediate powers of A
    # We do this reverse divide-and-conquer for efficiency reasons:
    # 1) it involves fewer padding steps for non-po2 L
    # 2) it involves more contiguous arrays

    # Take care of edge case for non-po2 arrays
    # Note that this initial step is a no-op for the case of power of 2 (l == L)
    k = v.shape[-1] - l
    v_ = powers.pop() @ v[..., l:]
    v = v[..., :l]
    v[..., :k] = v[..., :k] + v_

    # Handle reduction for power of 2
    while v.shape[-1] > 1:
        v = rearrange(v, '... (z l) -> ... z l', z=2)
        v = v[..., 0, :] + powers.pop() @ v[..., 1, :]
    return I, v.squeeze(-1)

def embed_c2r(A):
    A = rearrange(A, '... m n -> ... m () n ()')
    A = np.pad(A, ((0, 0), (0, 1), (0, 0), (0, 1))) + \
        np.pad(A, ((0, 0), (1, 0), (0, 0), (1,0)))
    return rearrange(A, 'm x n y -> (m x) (n y)')

def transition(measure, N, **measure_args):
    """ A, B transition matrices for different measures
    measure: the type of measure
      legt - Legendre (translated)
      legs - Legendre (scaled)
      glagt - generalized Laguerre (translated)
      lagt, tlagt - previous versions of (tilted) Laguerre with slightly different normalization
    """
    # Laguerre (translated)
    if measure == 'lagt':
        b = measure_args.get('beta', 1.0)
        A = np.eye(N) / 2 - np.tril(np.ones((N, N)))
        B = b * np.ones((N, 1))
    # Generalized Laguerre
    # alpha 0, beta small is most stable (limits to the 'lagt' measure)
    # alpha 0, beta 1 has transition matrix A = [lower triangular 1]
    elif measure == 'glagt':
        alpha = measure_args.get('alpha', 0.0)
        beta = measure_args.get('beta', 0.01)
        A = -np.eye(N) * (1 + beta) / 2 - np.tril(np.ones((N, N)), -1)
        B = ss.binom(alpha + np.arange(N), np.arange(N))[:, None]

        L = np.exp(.5 * (ss.gammaln(np.arange(N)+alpha+1) - ss.gammaln(np.arange(N)+1)))
        A = (1./L[:, None]) * A * L[None, :]
        B = (1./L[:, None]) * B * np.exp(-.5 * ss.gammaln(1-alpha)) * beta**((1-alpha)/2)
    # Legendre (translated)
    elif measure == 'legt':
        Q = np.arange(N, dtype=np.float64)
        R = (2*Q + 1) ** .5
        j, i = np.meshgrid(Q, Q)
        A = R[:, None] * np.where(i < j, (-1.)**(i-j), 1) * R[None, :]
        B = R[:, None]
        A = -A
    # Legendre (scaled)
    elif measure == 'legs':
        q = np.arange(N, dtype=np.float64)
        col, row = np.meshgrid(q, q)
        r = 2 * q + 1
        M = -(np.where(row >= col, r, 0) - np.diag(q))
        T = np.sqrt(np.diag(2 * q + 1))
        A = T @ M @ np.linalg.inv(T)
        B = np.diag(T)[:, None]
        B = B.copy() # Otherwise "UserWarning: given NumPY array is not writeable..." after t.as_tensor(B)
    elif measure == 'fourier':
        freqs = np.arange(N//2)
        d = np.stack([freqs, np.zeros(N//2)], axis=-1).reshape(-1)[:-1]
        A = 2*np.pi*(np.diag(d, 1) - np.diag(d, -1))
        A = A - embed_c2r(np.ones((N//2, N//2)))
        B = embed_c2r(np.ones((N//2, 1)))[..., :1]
    elif measure == 'random':
        A = np.random.randn(N, N) / N
        B = np.random.randn(N, 1)
    elif measure == 'diagonal':
        A = -np.diag(np.exp(np.random.randn(N)))
        B = np.random.randn(N, 1)
    else:
        raise NotImplementedError

    return A, B

def rank_correction(measure, N, rank=1, dtype=tf.dtypes.float32):
    """ Return low-rank matrix L such that A + L is normal """

    if measure == 'legs':
        assert rank >= 1
        P = tf.sqrt(.5+tf.range(N, dtype=dtype))
        P = tf.expand_dims(P, axis=0) # (1 N)
    elif measure == 'legt':
        assert rank >= 2
        P = tf.concat(1+2*tf.range(N, dtype=dtype)) # (N)
        P0 = P.clone()
        P0[0::2] = 0.
        P1 = P.clone()
        P1[1::2] = 0.
        P = tf.stack([P0, P1], axis=0) # (2 N)
    elif measure == 'lagt':
        assert rank >= 1
        P = .5**.5 * tf.ones(1, N, dtype=dtype)
    elif measure == 'fourier':
        P = tf.ones(N, dtype=dtype) # (N)
        P0 = P.clone()
        P0[0::2] = 0.
        P1 = P.clone()
        P1[1::2] = 0.
        P = tf.stack([P0, P1], axis=0) # (2 N)
    else: raise NotImplementedError

    d = tf.shape(P)[0]
    if rank > d:
        P = tf.concat([P, tf.zeros(rank-d, N, dtype=dtype)], dim=0) # (rank N)
    return P


def nplr(measure, N, rank=1, dtype=tf.dtypes.float32):
    """ Return w, p, q, V, B such that
    (w - p q^*, B) is unitarily equivalent to the original HiPPO A, B by the matrix V
    i.e. A = V[w - p q^*]V^*, B = V B
    """
    assert dtype == tf.dtypes.float32 or tf.dtypes.complex64
    if measure == 'random':
        dtype = tf.dtypes.complex64 if dtype == tf.dtypes.float32 else tf.dtypes.complex128
        # w = tf.random.normal(N//2, dtype=dtype)
        w = -tf.math.exp(tf.random.normal(N//2)) + 1j*tf.random.normal(N//2)
        P = tf.random.normal(rank, N//2, dtype=dtype)
        B = tf.random.normal(N//2, dtype=dtype)
        V = tf.eye(N, dtype=dtype)[..., :N//2] # Only used in testing
        return w, P, B, V

    A, B = transition(measure, N)
    A = tf.convert_to_tensor(A, dtype=dtype) # (N, N)
    B = tf.convert_to_tensor(B, dtype=dtype)[:, 0] # (N,)

    P = rank_correction(measure, N, rank=rank, dtype=dtype)
    P_2 = tf.expand_dims(P, axis=-2)
    P_1 = tf.expand_dims(P, axis=-1)
    AP = A + tf.math.reduce_sum(P_2 * P_1, axis=-3)
    w, V = tf.linalg.eig(AP) # (..., N) (..., N, N)
    # V w V^{-1} = A

    # Only keep one of the conjugate pairs
    # print('shape of w before: ', tf.shape(w)[0])
    w = w[..., 0::2]#.contiguous()
    # print('shape of w after: ', tf.shape(w)[0])
    V = V[..., 0::2]#.contiguous()
    # print('shape of V after: ', tf.shape(V))

    # V_inv = V.conj().transpose(-1, -2)
    # V_inv = tf.math.conj(V)
    V_inv = tf.transpose(V, perm=[1, 0], conjugate=True)


    B = contract('ij, j -> i', V_inv, tf.cast(B, V.dtype)) # V^* B
    P = contract('ij, ...j -> ...i', V_inv, tf.cast(P, V.dtype)) # V^* P


    return w, P, B, V






_c2r = lambda x: tf.stack( [tf.math.real(x), tf.math.imag(x)], axis=-1 )
_r2c = lambda x: tf.complex( x[...,0], x[...,1] )
_conj = lambda x: tf.concat( [x, tf.math.conj(x)], axis=-1 )
_resolve_conj = lambda x: tf.math.conj(x)
# v, w: (..., N); z: (..., L); returns: (..., L)
_cauchy_slow = lambda v,z,w: tf.reduce_sum( ( tf.expand_dims(v,axis=-1) /
                                              ( tf.expand_dims(z,axis=-2)-tf.expand_dims(w,axis=-1) ) ), axis=-2 )

class SSKernelNPLR(keras.layers.Layer):
    """Stores a representation of and computes the SSKernel function K_L(A^dt, B^dt, C) corresponding to a discretized state space, where A is Normal + Low Rank (NPLR)
    The class name stands for 'State-Space SSKernel for Normal Plus Low-Rank'.
    The parameters of this function are as follows.
    A: (... N N) the state matrix
    B: (... N) input matrix
    C: (... N) output matrix
    dt: (...) timescales / discretization step size
    p, q: (... P N) low-rank correction to A, such that Ap=A+pq^T is a normal matrix
    The forward pass of this Module returns:
    (... L) that represents represents FFT SSKernel_L(A^dt, B^dt, C)
    """

    # @t.no_grad()
    def _setup_C(self, double_length=False):
        """ Construct C~ from C
        double_length: current C is for length L, convert it to length 2L
        """
        C = _r2c(self.C)
        self._setup_state()
        dA_L = power(self.L, self.dA)
        # Multiply C by I - dA_L
        C_ = _conj(C)
        # print('S4Model | _setup_C | dA_L.shape:', dA_L.shape)
        prod = contract("h m n, c h n -> c h m", tf.transpose(dA_L, perm=[0,2,1]), C_)
        if double_length: prod = -prod # Multiply by I + dA_L instead
        C_ = C_ - prod
        C_ = C_[..., :self.N] # Take conjugate pairs again

        self.C = tf.Variable(_c2r(C_))

        if double_length:
            self.L *= 2
            self._omega(self.L, dtype=C.dtype, device=C.device, cache=True)

    def _omega(self, L, dtype, device, cache=True):
        """ Calculate (and cache) FFT nodes and their "unprocessed" them with the bilinear transform
        This should be called everytime the internal length self.L changes """
        omega_x = tf.convert_to_tensor(np.exp(-2j * np.pi / (L)), dtype=dtype)  # \omega_{2L}
        # print('omega_x:', omega_x)
        omega_y = tf.convert_to_tensor(np.arange(0, L // 2 + 1), dtype=dtype)
        # print('omega_y:', omega_y)
        omega = omega_x ** omega_y
        z = 2 * (1 - omega) / (1 + omega)
        self.omega = tf.Variable(_c2r(omega), trainable=False)
        self.z = tf.Variable(_c2r(z), trainable=False)
        return omega, z

    def __init__(
        self,
        L, w, P, B, C, log_dt,
        hurwitz=False,
        trainable=None,
        lr=None,
        tie_state=False,
        length_correction=True,
        verbose=False,
    ):
        """
        L: Maximum length; this module computes an SSM kernel of length L
        w: (N)
        p: (r, N) low-rank correction to A
        q: (r, N)
        A represented by diag(w) - pq^*
        B: (N)
        dt: (H) timescale per feature
        C: (H, C, N) system is 1-D to c-D (channels)
        hurwitz: tie pq and ensure w has negative real part
        trainable: toggle which of the parameters is trainable
        lr: add hook to set lr of hippo parameters specially (everything besides C)
        tie_state: tie all state parameters across the H hidden features
        length_correction: multiply C by (I - dA^L) - can be turned off when L is large for slight speedup at initialization (only relevant when N large as well)
        Note: tensor shape N here denotes half the true state size, because of conjugate symmetry
        """

        super().__init__()
        self.hurwitz = hurwitz
        self.tie_state = tie_state
        self.verbose = verbose

        # Rank of low-rank correction
        self.rank = P.shape[-2]
        assert w.shape[-1] == P.shape[-1] == B.shape[-1] == C.shape[-1]
        self.H = log_dt.shape[-1]
        self.N = w.shape[-1]

        # Broadcast everything to correct shapes
        correct_shapes = tf.broadcast_dynamic_shape(C.shape, (1, self.H, self.N))
        C = tf.broadcast_to(C, correct_shapes) # (H, C, N)
        H = 1 if self.tie_state else self.H
        B = repeat(B, 'n -> 1 h n', h=H)
        P = repeat(P, 'r n -> r h n', h=H)
        w = repeat(w, 'n -> h n', h=H)

        # Cache Fourier nodes every time we set up a desired length
        self.L = L
        if self.L is not None:
            self._omega(self.L, dtype=C.dtype, device=C.device, cache=True)

        # Register parameters
        # C is a regular parameter, not state
        # self.C = nn.Parameter(_c2r(C.conj().resolve_conj()))
        self.C = tf.Variable(_c2r(_resolve_conj(C)))
        train = False
        if trainable is None: trainable = {}
        if trainable == False: trainable = {}
        if trainable == True: trainable, train = {}, True
        self.log_dt = tf.Variable(log_dt, trainable=False)
        self.B = tf.Variable(_c2r(B), trainable=False)
        self.P = tf.Variable(_c2r(P), trainable=False)
        if self.hurwitz:
            log_w_real = tf.math.log(-w.real + 1e-3) # Some of the HiPPO methods have real part 0
            w_imag = w.imag
            self.log_w_real = tf.Variable(log_w_real, trainable=False)
            self.w_imag = tf.Variable(w_imag, trainable=False)
            self.Q = None
        else:
            self.w = tf.Variable(_c2r(w), trainable=False)
            Q = _resolve_conj( tf.identity(P) )
            self.Q = tf.Variable(_c2r(Q), trainable=False)

        if length_correction:
            self._setup_C()

    def _w(self):
        # Get the internal w (diagonal) parameter
        if self.hurwitz:
            w_real = -tf.math.exp(self.log_w_real)
            w_imag = self.w_imag
            w = w_real + 1j * w_imag
        else:
            w = _r2c(self.w)  # (..., N)
        return w

    def call(self, state=None, rate=1.0, L=None):
        """
        state: (..., s, N) extra tensor that augments B
        rate: sampling rate factor
        returns: (..., c+s, L)
        """
        # Handle sampling rate logic
        # The idea is that this kernel's length (in continuous units) is self.L, while we are asked to provide a kernel of length L at (relative) sampling rate rate
        # If either are not passed in, assume we're not asked to change the scale of our kernel
        assert not (rate is None and L is None)
        if rate is None:
            rate = self.L / L
        if L is None:
            L = int(self.L / rate)

        # Increase the internal length if needed
        # while rate * L > self.L:
        #     self.double_length()

        dt = tf.math.exp(self.log_dt) * rate
        B = _r2c(self.B)
        C = _r2c(self.C)
        P = _r2c(self.P)
        Q = P.conj() if self.Q is None else _r2c(self.Q)
        w = self._w()

        if rate == 1.0:
            # Use cached FFT nodes
            omega, z = _r2c(self.omega), _r2c(self.z)  # (..., L)
        else:
            omega, z = self._omega(int(self.L/rate), dtype=w.dtype, device=w.device, cache=False)

        if self.tie_state:
            B = repeat(B, '... 1 n -> ... h n', h=self.H)
            P = repeat(P, '... 1 n -> ... h n', h=self.H)
            Q = repeat(Q, '... 1 n -> ... h n', h=self.H)

        # Augment B
        if state is not None:
            # Have to "unbilinear" the state to put it into the same "type" as B
            # Compute 1/dt * (I + dt/2 A) @ state

            # Can do this without expanding (maybe minor speedup using conj symmetry in theory), but it's easier to read this way
            s = _conj(state) if state.shape[-1] == self.N else state # (B H N)
            sA = (
                s * _conj(w) # (B H N)
                - contract('bhm, rhm, rhn -> bhn', s, _conj(Q), _conj(P))
            )
            s = s / tf.expand_dims(dt, axis=-1) + sA / 2
            s = s[..., :self.N]

            B = tf.concat([s, B], dim=-3)  # (s+1, H, N)

        # Incorporate dt into A
        w = w * tf.cast(tf.expand_dims(dt, axis=-1), dtype=w.dtype)  # (H N)

        # Stack B and p, C and q for convenient batching
        B = tf.concat([B, P], axis=-3) # (s+1+r, H, N)
        C = tf.concat([C, Q], axis=-3) # (c+r, H, N)

        # Incorporate B and C batch dimensions
        v = tf.expand_dims(B, axis=-3) * tf.expand_dims(C, axis=-4)  # (s+1+r, c+r, H, N)
        # w = w[None, None, ...]  # (1, 1, H, N)
        # z = z[None, None, None, ...]  # (1, 1, 1, L)

        # Calculate resolvent at omega
        # if has_cauchy_extension and z.dtype == tf.dtypes.complex64:
        #     r = cauchy_mult(v, z, w, symmetric=True)
        # elif has_pykeops:
        #     r = cauchy_conj(v, z, w)
        # else:
        #     r = cauchy_slow(v, z, w)
        r = _cauchy_slow(v, z, w)
        # print('S4Model | SSKernelNPLR | r:', r)
        # print('S4Model | SSKernelNPLR | dt:', dt)
        r = r * tf.cast( tf.reshape(dt, shape=[1,1,-1,1]), r.dtype )  # (S+1+R, C+R, H, L)

        # Low-rank Woodbury correction
        if self.rank == 1:
            k_f = r[:-1, :-1, :, :] - r[:-1, -1:, :, :] * r[-1:, :-1, :, :] / (1 + r[-1:, -1:, :, :])
        elif self.rank == 2:
            r00 = r[: -self.rank, : -self.rank, :, :]
            r01 = r[: -self.rank, -self.rank :, :, :]
            r10 = r[-self.rank :, : -self.rank, :, :]
            r11 = r[-self.rank :, -self.rank :, :, :]
            det = (1 + r11[:1, :1, :, :]) * (1 + r11[1:, 1:, :, :]) - r11[:1, 1:, :, :] * r11[1:, :1, :, :]
            s = (
                r01[:, :1, :, :] * (1 + r11[1:, 1:, :, :]) * r10[:1, :, :, :]
                + r01[:, 1:, :, :] * (1 + r11[:1, :1, :, :]) * r10[1:, :, :, :]
                - r01[:, :1, :, :] * (r11[:1, 1:, :, :]) * r10[1:, :, :, :]
                - r01[:, 1:, :, :] * (r11[1:, :1, :, :]) * r10[:1, :, :, :]
            )
            s = s / det
            k_f = r00 - s
        else:
            r00 = r[:-self.rank, :-self.rank, :, :]
            r01 = r[:-self.rank, -self.rank:, :, :]
            r10 = r[-self.rank:, :-self.rank, :, :]
            r11 = r[-self.rank:, -self.rank:, :, :]
            r11 = rearrange(r11, "a b h n -> h n a b")
            r11 = tf.linalg.inv(tf.eye(self.rank, device=r.device) + r11)
            r11 = rearrange(r11, "h n a b -> a b h n")
            k_f = r00 - tf.einsum("i j h n, j k h n, k l h n -> i l h n", r01, r11, r10)

        # Final correction for the bilinear transform
        k_f = k_f * 2 / (1 + omega)

        # Move from frequency to coefficients
        k = tf.signal.irfft(k_f)  # (S+1, C, H, L)

        # Truncate to target length
        k = k[..., :L]

        if state is not None:
            k_state = k[:-1, :, :, :]  # (S, C, H, L)
        else:
            k_state = None
        k_B = k[-1, :, :, :] # (C H L)
        return k_B, k_state

    # @t.no_grad()
    def double_length(self):
        if self.verbose: log.info(f"S4: Doubling length from L = {self.L} to {2*self.L}")
        self._setup_C(double_length=True)

    def _setup_linear(self):
        """ Create parameters that allow fast linear stepping of state """
        w = self._w()
        B = _r2c(self.B) # (H N)
        P = _r2c(self.P)
        Q = P.conj() if self.Q is None else _r2c(self.Q)

        # Prepare Linear stepping
        dt = tf.math.exp(self.log_dt)
        D = tf.math.reciprocal( tf.cast(2.0 / tf.expand_dims(dt, axis=-1), dtype=w.dtype) - w)  # (H, N)
        # R = ( tf.eye(self.rank, dtype=w.dtype) + 2* tf.math.real(contract('r h n, h n, s h n -> h r s', Q, D, P)) ) # (H r r)
        R = ( tf.eye(self.rank) + 2* tf.math.real(contract('r h n, h n, s h n -> h r s', Q, D, P)) ) # (H r r)
        Q_D = rearrange(Q*D, 'r h n -> h r n')
        R = tf.linalg.solve(tf.cast(R, dtype=Q_D.dtype), Q_D) # (H r N)
        R = rearrange(R, 'h r n -> r h n')
        E = tf.cast(2.0 / tf.expand_dims(dt, axis=-1), dtype=w.dtype) + w

        self.step_params = {
            "D": D, # (H N)
            "R": R, # (r H N)
            "P": P, # (r H N)
            "Q": Q, # (r H N)
            "B": B, # (1 H N)
            "E": E, # (H N)
        }

    def _step_state_linear(self, u=None, state=None):
        """
        Version of the step function that has time O(N) instead of O(N^2) per step, which takes advantage of the DPLR form and bilinear discretization.
        Unfortunately, as currently implemented it's about 2x slower because it calls several sequential operations. Perhaps a fused CUDA kernel implementation would be much faster
        u: (H) input
        state: (H, N/2) state with conjugate pairs
          Optionally, the state can have last dimension N
        Returns: same shape as state
        """
        C = _r2c(self.C) # View used for dtype/device

        if u is None: # Special case used to find dA
            u = tf.zeros(self.H, dtype=C.dtype)
        if state is None: # Special case used to find dB
            state = tf.zeros([self.H, self.N], dtype=C.dtype)

        step_params = self.step_params.copy()
        if state.shape[-1] == self.N: # Only store half of the conjugate pairs; should be true by default
            # There should be a slightly faster way using conjugate symmetry
            contract_fn = lambda p, x, y: contract('r h n, r h m, ... h m -> ... h n', _conj(p), _conj(x), _conj(y))[..., :self.N] # inner outer product
            contract_fn1 = contract_fn
        else:
            assert state.shape[-1] == 2*self.N
            step_params = {k: _conj(v) for k, v in step_params.items()}
            # TODO worth setting up a contract_expression in default_state if we want to use this at inference time for stepping
            contract_fn = lambda p, x, y: contract('r h n, r h m, ... h m -> ... h n', p, x, y) # inner outer product
            contract_fn1 = lambda p, x, y: contract('r h n, r h m, ... 1 m -> ... h n', p, x, y)  # inner outer product
        D = step_params["D"]  # (H N)
        E = step_params["E"]  # (H N)
        R = step_params["R"]  # (r H N)
        P = step_params["P"]  # (r H N)
        Q = step_params["Q"]  # (r H N)
        B = step_params["B"]  # (1 H N)

        new_state = E * state - contract_fn1(P, Q, state) # (B H N)
        new_state = new_state + 2.0 * B * tf.expand_dims(u, axis=-1)  # (B H N)
        new_state = D * (new_state - contract_fn(P, R, new_state))

        return new_state

    def _setup_state(self):
        """ Construct dA and dB for discretized state equation """

        # Construct dA and dB by using the stepping
        self._setup_linear()
        C = _r2c(self.C) # Just returns a view that we use for finding dtype/device

        state = tf.expand_dims( tf.eye(2*self.N, dtype=C.dtype) , axis=-2)   # (N 1 N)
        dA = self._step_state_linear(state=state)
        dA = rearrange(dA, "n h m -> h m n")
        self.dA = dA # (H N N)

        u = tf.ones(self.H, dtype=C.dtype)
        dB = self._step_state_linear(u=u)
        dB = _conj(dB)
        self.dB = rearrange(dB, '1 h n -> h n') # (H N)

    def _step_state(self, u, state):
        """ Must be called after self.default_state() is used to construct an initial state!  """
        next_state = self.state_contraction(self.dA, state) + self.input_contraction(self.dB, u)
        return next_state


    def setup_step(self, mode='dense'):
        """ Set up dA, dB, dC discretized parameters for stepping """
        self._setup_state()

        # Calculate original C
        dA_L = power(self.L, self.dA)
        I = tf.eye(self.dA.shape[-1]).to(dA_L)
        C = _conj(_r2c(self.C)) # (H C N)

        dC = tf.linalg.solve(
            I - dA_L.transpose(-1, -2),
            tf.expand_dims(C, axis=-1),
        ).squeeze(-1)
        self.dC = dC

        # Do special preprocessing for different step modes

        self._step_mode = mode
        if mode == 'linear':
            # Linear case: special step function for the state, we need to handle output
            # use conjugate symmetry by default, which affects the output projection
            self.dC = 2*self.dC[:, :, :self.N]
        elif mode == 'diagonal':
            # Eigendecomposition of the A matrix
            L, V = tf.linalg.eig(self.dA)
            V_inv = tf.linalg.inv(V)
            # Check that the eigendedecomposition is correct
            if self.verbose:
                print("Diagonalization error:", torch.dist(V @ torch.diag_embed(L) @ V_inv, self.dA))

            # Change the parameterization to diagonalize
            self.dA = L
            self.dB = contract('h n m, h m -> h n', V_inv, self.dB)
            self.dC = contract('h n m, c h n -> c h m', V, self.dC)

        elif mode == 'dense':
            pass
        else: raise NotImplementedError("NPLR Kernel step mode must be {'dense' | 'linear' | 'diagonal'}")


    def default_state(self, *batch_shape):
        C = _r2c(self.C)
        N = C.shape[-1]
        H = C.shape[-2]

        # Cache the tensor contractions we will later do, for efficiency
        # These are put in this function because they depend on the batch size
        if self._step_mode !='linear':
            N *= 2

            if self._step_mode == 'diagonal':
                self.state_contraction = contract_expression(
                    "h n, ... h n -> ... h n",
                    (H, N),
                    batch_shape + (H, N),
                )
            else:
                # Dense (quadratic) case: expand all terms
                self.state_contraction = contract_expression(
                    "h m n, ... h n -> ... h m",
                    (H, N, N),
                    batch_shape + (H, N),
                )

            self.input_contraction = contract_expression(
                "h n, ... h -> ... h n",
                (H, N), # self.dB.shape
                batch_shape + (H,),
            )

        self.output_contraction = contract_expression(
            "c h n, ... h n -> ... c h",
            (C.shape[0], H, N), # self.dC.shape
            batch_shape + (H, N),
        )

        state = tf.zeros(*batch_shape, H, N, dtype=C.dtype, device=C.device)
        return state

    def step(self, u, state):
        """ Must have called self.setup_step() and created state with self.default_state() before calling this """

        if self._step_mode == 'linear':
            new_state = self._step_state_linear(u, state)
        else:
            new_state = self._step_state(u, state)
        y = self.output_contraction(self.dC, new_state)
        return y, new_state





class HippoSSKernel(keras.layers.Layer):
    """Wrapper around SSKernel that generates A, B, C, dt according to HiPPO arguments.
    The SSKernel is expected to support the interface
    forward()
    default_state()
    setup_step()
    step()
    """

    def __init__(
            self,
            H,
            N=64,
            L=1,
            measure="legs",
            rank=1,
            channels=1,  # 1-dim to C-dim map; can think of C as having separate "heads"
            dt_min=0.001,
            dt_max=0.1,
            trainable=None,  # Dictionary of options to train various HiPPO parameters
            lr=None,  # Hook to set LR of hippo parameters differently
            length_correction=True,
            # Multiply by I-A|^L after initialization; can be turned off for initialization speed
            hurwitz=False,
            tie_state=False,  # Tie parameters of HiPPO ODE across the H features
            precision=1,  # 1 (single) or 2 (double) for the kernel
            resample=False,
            # If given inputs of different lengths, adjust the sampling rate. Note that L should always be provided in this case, as it assumes that L is the true underlying length of the continuous signal
            verbose=False,
    ):
        super().__init__()
        self.N = N
        self.H = H
        L = L or 1
        self.precision = precision
        dtype = tf.dtypes.double if self.precision == 2 else tf.dtypes.float32
        cdtype = tf.dtypes.complex64 if dtype == tf.dtypes.float32 else tf.dtypes.complex128
        self.rate = None if resample else 1.0
        self.channels = channels

        # Generate dt
        # print('S4Model/HippoSSKernel: H=', self.H)
        log_dt = tf.random.uniform([self.H], dtype=dtype) * (
                math.log(dt_max) - math.log(dt_min)
        ) + math.log(dt_min)

        w, p, B, _ = nplr(measure, self.N, rank, dtype=dtype)
        # C = tf.random.normal(channels, self.H, self.N // 2, dtype=cdtype)
        C = tf.complex(tf.random.normal([channels, self.H, self.N // 2]), tf.random.normal([channels, self.H, self.N // 2]))

        self.kernel = SSKernelNPLR(
            L, w, p, B, C,
            log_dt,
            hurwitz=hurwitz,
            trainable=trainable,
            lr=lr,
            tie_state=tie_state,
            length_correction=length_correction,
            verbose=verbose,
        )

    def call(self, L=None):
        k, _ = self.kernel(state=None, rate=self.rate, L=L)
        return tf.cast(k, dtype=tf.float32)

    def step(self, u, state, **kwargs):
        u, state = self.kernel.step(u, state, **kwargs)
        return tf.cast(u, dtype=tf.float32), state

    def default_state(self, *args, **kwargs):
        return self.kernel.default_state(*args, **kwargs)


class Identity(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Identity, self).__init__(**kwargs)
    def call(self, x):
        return x
def Activation(activation=None, dim=-1):
    if activation in [ None, 'id', 'identity', 'linear' ]:
        return Identity()
    elif activation == 'tanh':
        return keras.layers.Activation('tanh')
    elif activation == 'relu':
        return keras.layers.Activation('relu')
    elif activation == 'gelu':
        return keras.layers.Activation('gelu')
    elif activation in ['swish', 'silu']:
        keras.layers.Activation('swish')
    elif activation == 'glu':
        return lambda x: tf.multiply( tf.split(split_dim=dim,num_split=2,value=x)[0],
                                      tf.sigmoid( tf.split(split_dim=dim,num_split=2,value=x)[1] ) )
    elif activation == 'sigmoid':
        return keras.layers.Activation('sigmoid')
    else:
        raise NotImplementedError("hidden activation '{}' is not implemented".format(activation))



def get_initializer(name, activation=None):
    if activation in [ None, 'id', 'identity', 'linear', 'modrelu' ]:
        nonlinearity = 'linear'
    elif activation in ['relu', 'tanh', 'sigmoid']:
        nonlinearity = activation
    elif activation in ['gelu', 'swish', 'silu']:
        nonlinearity = 'relu' # Close to ReLU so approximate with ReLU's gain
    else:
        raise NotImplementedError(f"get_initializer: activation {activation} not supported")

    if name == 'uniform':
        initializer = partial(torch.nn.init.kaiming_uniform_, nonlinearity=nonlinearity)
    elif name == 'normal':
        initializer = partial(torch.nn.init.kaiming_normal_, nonlinearity=nonlinearity)
    elif name == 'xavier':
        initializer = torch.nn.init.xavier_normal_
    elif name == 'zero':
        initializer = partial(torch.nn.init.constant_, val=0)
    elif name == 'one':
        initializer = partial(torch.nn.init.constant_, val=1)
    else:
        raise NotImplementedError(f"get_initializer: initializer type {name} not supported")

    return initializer

class TransposedLinear(keras.layers.Layer):
    """ Linear module on the second-to-last dimension """
    def __init__(self, d_input, d_output, use_bias=True):
        self.d_input = d_input
        self.d_output = d_output
        self.use_bias = use_bias
        super(TransposedLinear, self).__init__()
    def build(self, input_shape):
        weight_initializer = keras.initializers.HeUniform()
        self.weight = weight_initializer(shape=(self.d_output, self.d_input))
        if self.use_bias:
            bound = 1 / math.sqrt(self.d_input)
            bias_initializer = keras.initializers.random_uniform(minval=-bound, maxval=bound)
            self.bias = bias_initializer(shape=(self.d_output, 1))
        else:
            self.bias = 0.0
    def call(self, x):
        return contract('... u l, v u -> ... v l', x, self.weight) + self.bias

def LinearActivation(
        d_input, d_output, bias=True,
        zero_bias_init=False,
        transposed=False,
        initializer=None,
        activation=None,
        activate=False, # Apply activation as part of this module
        weight_norm=False,
        **kwargs,
    ):
    """ Returns a linear layer with control over axes order, initialization, and activation """

    # Construct core module
    linear_cls = TransposedLinear if transposed else keras.layers.Dense
    if activation == 'glu': d_output *= 2
    linear = linear_cls(d_input, d_output, use_bias=bias, **kwargs)

    # Initialize weight
    if initializer is not None:
        get_initializer(initializer, activation)(linear.weight)

    # Initialize bias
    if bias and zero_bias_init:
        nn.init.zeros_(linear.bias)

    # Weight norm
    if weight_norm:
        linear = nn.utils.weight_norm(linear)

    if activate and activation is not None:
        activation = Activation(activation, dim=-2 if transposed else -1)
        linear = nn.Sequential(linear, activation)
    return linear


class S4(keras.layers.Layer):
    def __init__(
            self,
            d_model,
            d_state=64,
            l_max=1,
            # Maximum length of sequence. Fine if not provided: the kernel will keep doubling in length until longer than sequence. However, this can be marginally slower if the true length is not a power of 2
            channels=1,  # maps 1-dim to C-dim
            bidirectional=False,
            # Arguments for FF
            activation='gelu',  # activation in between SS and FF
            postact=None,  # activation after FF
            initializer=None,  # initializer on FF
            weight_norm=False,  # weight normalization on FF
            hyper_act=None,  # Use a "hypernetwork" multiplication
            dropout=0.0,
            transposed=True,  # axis ordering (B, L, D) or (B, D, L)
            verbose=False,
            # SSM Kernel arguments
            **kernel_args,
    ):

        """
        d_state: the dimension of the state, also denoted by N
        l_max: the maximum sequence length, also denoted by L
          if this is not known at model creation, set l_max=1
        channels: can be interpreted as a number of "heads"
        bidirectional: bidirectional
        dropout: standard dropout argument
        transposed: choose backbone axis ordering of (B, L, H) or (B, H, L) [B=batch size, L=sequence length, H=hidden dimension]

        Other options are all experimental and should not need to be configured
        """

        super().__init__()
        if verbose:
            import src.utils.train
            log = src.utils.train.get_logger(__name__)
            log.info(f"Constructing S4 (H, N, L) = ({d_model}, {d_state}, {l_max})")

        self.h = d_model
        self.n = d_state
        self.bidirectional = bidirectional
        self.channels = channels
        self.transposed = transposed

        # optional multiplicative modulation GLU-style
        # https://arxiv.org/abs/2002.05202
        self.hyper = hyper_act is not None
        if self.hyper:
            channels *= 2
            self.hyper_activation = Activation(hyper_act)

        self.D = tf.Variable(tf.random.normal([channels, self.h]))

        if self.bidirectional:
            channels *= 2

        # SSM Kernel
        # print('S4Model/S4: h=', self.h)
        self.kernel = HippoSSKernel(self.h, N=self.n, L=l_max, channels=channels, verbose=verbose, **kernel_args)

        # Pointwise
        self.activation = Activation(activation)
        dropout_fn = keras.layers.SpatialDropout2D if self.transposed else keras.layers.Dropout
        self.dropout = dropout_fn(dropout) if dropout > 0.0 else Identity()

        # position-wise output transform to mix features
        self.output_linear = LinearActivation(
            self.h * self.channels,
            self.h,
            transposed=self.transposed,
            initializer=initializer,
            activation=postact,
            activate=True,
            weight_norm=weight_norm,
        )

        # self.time_transformer = get_torch_trans(heads=8, layers=1, channels=self.h)

    def call(self, u, **kwargs):  # absorbs return_output and transformer src mask
        """
        u: (B H L) if self.transposed else (B L H)
        state: (H N) never needed unless you know what you're doing

        Returns: same shape as u
        """
        # print('S4Model | S4 | u.shape: ', u.shape)
        # print('S4Model | S4 | self.h: ', self.h)

        if not self.transposed: u = u.transpose(-1, -2)
        L = u.shape[-1]

        # Compute SS Kernel
        # print('S4Model | S4 | L: ', L)
        k = self.kernel(L=L)  # (C H L) (B C H L)

        # Convolution
        if self.bidirectional:
            # k0, k1 = rearrange(k, '(s c) h l -> s c h l', s=2)
            # print('S4Model | S4 | k.shape', k.shape)
            k0, k1 = tf.expand_dims(k[0],axis=0), tf.expand_dims(k[1],axis=0)
            # print('S4Model | S4 | k0.shape', k0.shape)
            # print('S4Model | S4 | k1.shape', k1.shape)
            pad0 = np.zeros([3,2])
            pad0[2,1] = L
            pad1 = np.zeros([3,2])
            pad1[2,1] = L
            # k = F.pad(k0, (0, L)) + F.pad(k1.flip(-1), (L, 0))
            k = tf.pad(k0, paddings=pad0) + tf.pad(tf.reverse(k1, axis=[-1]), paddings=pad1)
            # print('S4Model | S4 | k.shape', k.shape)

        # print('S4Model | S4 | L', L)
        # k_f = torch.fft.rfft(k, n=2 * L)  # (C H L)
        k_f = tf.signal.rfft( input_tensor=k, fft_length=tf.constant(value=[2*L]) ) # (C H L)
        # u_f = torch.fft.rfft(u, n=2 * L)  # (B H L)
        u_f = tf.signal.rfft( input_tensor=u, fft_length=tf.constant(value=[2*L]) ) # (B H L)

        y_f = contract('bhl,chl->bchl', u_f, k_f) # (B C H L)
        # y = tf.signal.irfft(y_f, n=2 * L)[..., :L]  # (B C H L)
        # print('S4Model | S4 | y_f.shape:', y_f.shape)
        y = tf.signal.irfft(y_f, fft_length=tf.constant(value=[2*L]))[..., :L]  # (B C H L)

        # Compute D term in state space equation - essentially a skip connection
        y = y + contract('bhl,ch->bchl', u, self.D)

        # Optional hyper-network multiplication
        if self.hyper:
            y, yh = rearrange(y, 'b (s c) h l -> s b c h l', s=2)
            y = self.hyper_activation(yh) * y

        # Reshape to flatten channels
        y = rearrange(y, '... c h l -> ... (c h) l')

        y = self.dropout(self.activation(y))

        if not self.transposed: y = y.transpose(-1, -2)

        y = self.output_linear(y)

        # ysize = b, k, l, requieres l, b, k
        # y = self.time_transformer(y.permute(2,0,1)).permute(1,2,0)

        return y, None

    def step(self, u, state):
        """ Step one time step as a recurrent model. Intended to be used during validation.

        u: (B H)
        state: (B H N)
        Returns: output (B H), state (B H N)
        """
        assert not self.training

        y, next_state = self.kernel.step(u, state)  # (B C H)
        u_2 = tf.expand_dims(u, axis=-2)
        y = y + u_2 * self.D
        y = rearrange(y, '... c h -> ... (c h)')
        y = self.activation(y)
        if self.transposed:
            y_1 = tf.expand(y, axis=-1)
            y = self.output_linear( y_1 )
            y = tf.squeeze(y, axis=-1) # .squeeze(-1)
        else:
            y = self.output_linear(y)
        return y, next_state

    def default_state(self, *batch_shape, device=None):
        return self.kernel.default_state(*batch_shape)

    @property
    def d_state(self):
        return self.h * self.n

    @property
    def d_output(self):
        return self.h

    @property
    def state_to_tensor(self):
        return lambda state: rearrange('... h n -> ... (h n)', state)


class S4Layer(keras.layers.Layer):
    # S4 Layer that can be used as a drop-in replacement for a TransformerEncoder
    def __init__(self, features, lmax, N=64, dropout=0.0, bidirectional=True, layer_norm=True):
        super().__init__()
        self.s4_layer = S4(d_model=features,
                            d_state=N,
                            l_max=lmax,
                            bidirectional=bidirectional)
        # print('S4Model | S4Layer | features:', features)
        self.norm_layer = keras.layers.LayerNormalization(axis=-1) if layer_norm else Identity()
        self.dropout = keras.layers.SpatialDropout2D(dropout) if dropout >0 else Identity()

    def call(self, x):
        # x has shape seq, batch, feature
        x = tf.transpose(x, perm=[1,2,0])  # batch, feature, seq (as expected from S4 with transposed=True)
        xout, _ = self.s4_layer(x)  # batch, feature, seq
        xout = self.dropout(xout)
        xout = xout + x # skip connection   # batch, feature, seq
        xout = tf.transpose(xout, perm=[2,0,1]) # seq, batch, feature
        # print('S4Model | S4Layer | xout.shape:', xout.shape)
        return self.norm_layer(xout)