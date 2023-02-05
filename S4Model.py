from tensorflow import keras

class S4Layer(keras.layers.Layer):
    # S4 Layer that can be used as a drop-in replacement for a TransformerEncoder
    def __init__(self, features, lmax, N=64, dropout=0.0, bidirectional=True, layer_norm=True):
        super().__init__()
        self.s4_layer  = S4(d_model=features,
                            d_state=N,
                            l_max=lmax,
                            bidirectional=bidirectional)

        self.norm_layer = nn.LayerNorm(features) if layer_norm else nn.Identity()
        self.dropout = nn.Dropout2d(dropout) if dropout >0 else nn.Identity()

    def forward(self, x):
        # x has shape seq, batch, feature
        x = x.permute((1 ,2 ,0))  # batch, feature, seq (as expected from S4 with transposed=True)
        xout, _ = self.s4_layer(x)  # batch, feature, seq
        xout = self.dropout(xout)
        xout = xout + x # skip connection   # batch, feature, seq
        xout = xout.permute((2 ,0 ,1)) # seq, batch, feature
        return self.norm_layer(xout)