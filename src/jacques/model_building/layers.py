from torch import nn
from torch.multiprocessing import set_start_method
try:
     set_start_method('spawn')
except RuntimeError:
    pass


class HeadNet():

    def __init__(self, bodynet_features_out, head_aggregation_function, head_hidden_layers_activation_function,
                 head_normalization_function, head_proba_dropout, nb_hidden_head_layers, nb_classes):
        self.bodynet_features_out = bodynet_features_out
        self.head_aggregation_function = head_aggregation_function
        self.head_hidden_layers_activation_function = head_hidden_layers_activation_function()
        self.head_normalization_function = head_normalization_function
        self.dropout = nn.Dropout(head_proba_dropout)
        self.nb_hidden_head_layers = nb_hidden_head_layers
        self.nb_classes = nb_classes

    def create_one_hidden_head_layer(self, nb_features_in, nb_features_out):
        hidden_head_layer = []
        hidden_head_layer.append(self.head_aggregation_function(nb_features_in, nb_features_out))
        hidden_head_layer.append(self.head_hidden_layers_activation_function)
        hidden_head_layer.append(self.head_normalization_function(nb_features_out))
        hidden_head_layer.append(self.dropout)
        return nn.Sequential(*hidden_head_layer)

    def create_output_layer(self, nb_features_last_hidden_layer):
        return nn.Sequential(self.head_aggregation_function(nb_features_last_hidden_layer, self.nb_classes))

    def create_head_layers(self):
        nb_features_headnet_hidden_layer_in = self.bodynet_features_out
        nb_features_hidden_layer_out = self.bodynet_features_out//2
        head_layers = []
        for hidden_layer in range(self.nb_hidden_head_layers):
            hidden_head_layer = self.create_one_hidden_head_layer(nb_features_headnet_hidden_layer_in, nb_features_hidden_layer_out)
            nb_features_headnet_hidden_layer_in //= 2
            nb_features_hidden_layer_out //= 2
            head_layers.append(hidden_head_layer)
        output_layer = self.create_output_layer(nb_features_hidden_layer_out*2)
        head_layers.append(output_layer)
        return nn.Sequential(*head_layers)

    
def build_model(backbone, headnet):
    # init a pretrained model & freeze the backbone parameters with requires_grad = False
    for param in backbone.parameters():
        param.requires_grad = False
        
    #replace the fc layer by head layers
    backbone.fc = headnet.create_head_layers()
    return backbone
    
    
