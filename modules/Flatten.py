class flatten():
    def __init__(self, input_dimensions=(1,1,1)):
        h, w, c = input_dimensions
        self.output_features = h*w*c
        self.input_dimensions = (-1, c, h, w)
    
    def forward(self, X):
        # X dims are (m, h, w, c)
        # returns X with dims (m, h*w*c)
        return X.transpose(0,3,1,2).reshape(-1, self.output_features).T

    def backward(self, dA):
        # dA dimensions are (number of features) x (number of examples)
        # returns dA with dims (m, h, w, c)
        return dA.T.reshape(self.input_dimensions).transpose(0, 2, 3, 1)
    
    def output_dims(self):
        return self.output_features
    def getGrads(self):
        return None

    def getLayerParams(self):
        LayerParams = self.output_features, self.input_dimensions
        return "flatten", LayerParams

    def setLayerParams(self, LayerParams):
        self.output_features, self.input_dimensions = LayerParams
        