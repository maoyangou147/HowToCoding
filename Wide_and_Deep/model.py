import torch
import torch.nn as nn

class Wide(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(in_features=input_dim, out_features=1)
    
    def forward(self, x):
        return self.linear(x)

class Deep(nn.Module):
    def __init__(self, config, hidden_layers):
        super().__init__()
        self.dnn = nn.ModuleList([nn.Linear(layer[0], layer[1]) for layer in list(zip(hidden_layers[:-1], hidden_layers[1:]))])
        self.dropout = nn.Dropout(p = config['deep_dropout'])
    
    def forward(self, x):
        for layer in self.dnn:
            x = layer(x)
            x = torch.relu(x)
        x = self.dropout(x)
        return x

class WideDeep(nn.Module):
    def __init__(self, config, dense_features_cols, sparse_features_cols):
        super().__init__()
        self._config = config
        self._num_of_dense_feature = len(dense_features_cols)
        self.sparse_features_cols = sparse_features_cols

        self.embedding_layers = nn.ModuleList(
            [nn.Embedding(num_embeddings=num_feat, embedding_dim=config['embed_dim']) for num_feat in self.sparse_features_cols]
        )

        # Deep hidden layers
        self._deep_hidden_layers = config['hidden_layers']
        self._deep_hidden_layers.insert(0, self._num_of_dense_feature + config['embed_dim'] * len(self.sparse_features_cols))

        self._wide = Wide(self._num_of_dense_feature)
        self._deep = Deep(config, self._deep_hidden_layers)
        # 之前直接将这个final_layer加入到了Deep模块里面，想着反正输出都是1，结果没注意到Deep没经过一个Linear层都会经过Relu激活函数，如果
        # 最后输出层大小是1的话，再经过ReLU之后，很可能变为了0，造成梯度消失问题，导致Loss怎么样都降不下来。
        self._final_linear = nn.Linear(self._deep_hidden_layers[-1], 1)
    
    def forward(self, x):
        # 先区分出稀疏特征和稠密特征，这里是按照列来划分的，即所有的行都要进行筛选
        dense_input, sparse_input = x[:, :self._num_of_dense_feature], x[:, self._num_of_dense_feature:]
        sparse_input = sparse_input.long()

        sparse_embeds = [self.embedding_layers[i](sparse_input[:, i]) for i in range(sparse_input.shape[1])]
        sparse_embeds = torch.cat(sparse_embeds, axis=-1)

        # Deep模块的输入是稠密特征和稀疏特征经过Embedding产生的稠密特征的
        deep_input = torch.cat([sparse_embeds, dense_input], dim=-1)

        wide_out = self._wide(dense_input)
        deep_out = self._deep(deep_input)
        deep_out = self._final_linear(deep_out)

        assert (wide_out.shape == deep_out.shape)

        outputs = torch.sigmoid(0.5 * (wide_out + deep_out))
        return outputs

    def saveModel(self):
        torch.save(self.state_dict(), self._config['model_name'])

    def loadModel(self, map_location):
        state_dict = torch.load(self._config['model_name'], map_location=map_location)
        self.load_state_dict(state_dict, strict=False)

