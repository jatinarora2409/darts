import numpy as np
import torch
from scipy.sparse import csc_matrix, csr_matrix
from sklearn.cluster import KMeans
from torch import nn

from model import Cell
from operations import ReLUConvBN

def apply_weight_sharing(model, bits=2):
    """
    Applies weight sharing to the given model
    """
    flag = False
    for modules in model.children():
        if not isinstance(modules, nn.AdaptiveAvgPool3d):
            for module in modules:
                if not isinstance(module, Cell):
                    dev = module.weight.device
                    weight = module.weight.data.cpu().numpy()
                    shape = weight.shape
                    result = weight.copy()
                    # flag = True
                    if len(shape) > 2:
                        for i in range(shape[0]):
                            for j in range(shape[1]):
                                wt = weight[i][j]
                                mat = csr_matrix(wt) if wt.shape[0] < wt.shape[1] else csc_matrix(wt)
                                min_ = min(mat.data)
                                max_ = max(mat.data)
                                space = np.linspace(min_, max_, num=2 ** bits)
                                kmeans = KMeans(n_clusters=len(space), init=space.reshape(-1, 1), n_init=1,
                                                precompute_distances=True, algorithm="full")
                                kmeans.fit(mat.data.reshape(-1, 1))
                                new_weight = kmeans.cluster_centers_[kmeans.labels_].reshape(-1)
                                mat.data = new_weight
                                result[i][j] = mat.toarray()

                        module.weight.data = torch.from_numpy(result).to(dev)
                else:
                    print(module)
                    for cell_module in module.children():
                        if isinstance(cell_module, ReLUConvBN):
                            for ReLU_List in cell_module.children():
                                for ReLU_module in ReLU_List:
                                    if isinstance(ReLU_module, nn.Conv2d):
                                        if ReLU_module.kernel_size[0] == 1:
                                            continue;
                                        dev = ReLU_module.weight.device
                                        weight = ReLU_module.weight.data.cpu().numpy()
                                        shape = weight.shape
                                        print(shape)
                                        result = weight.copy()
                                        # if len(shape) > 2:
                                        #     for i in range(shape[0]):
                                        #         for j in range(shape[1]):
                                        #             wt = weight[i][j]
                                        #             mat = csr_matrix(wt) if wt.shape[0] < wt.shape[1] else csc_matrix(wt)
                                        #             min_ = min(mat.data)
                                        #             max_ = max(mat.data)
                                        #             space = np.linspace(min_, max_, num=2 ** bits)
                                        #             kmeans = KMeans(n_clusters=len(space), init=space.reshape(-1, 1),
                                        #                             n_init=1,
                                        #                             precompute_distances=True, algorithm="full")
                                        #             kmeans.fit(mat.data.reshape(-1, 1))
                                        #             new_weight = kmeans.cluster_centers_[kmeans.labels_].reshape(-1)
                                        #             mat.data = new_weight
                                        #             result[i][j] = mat.toarray()
                                        #
                                        #     ReLU_module.weight.data = torch.from_numpy(result).to(dev)
                        elif isinstance(cell_module, nn.ModuleList):
                            for innerModules in cell_module.children():
                                for seqItem in innerModules.children():
                                    for layer in seqItem:
                                        if isinstance(layer, nn.Conv2d):
                                            if layer.kernel_size[0] == 1:
                                                continue;
                                            dev = layer.weight.device
                                            weight = layer.weight.data.cpu().numpy()
                                            shape = weight.shape
                                            print(shape)
                                            result = weight.copy()
                                            if len(shape) > 2:
                                                for i in range(shape[0]):
                                                    for j in range(shape[1]):
                                                        wt = weight[i][j]
                                                        mat = csr_matrix(wt) if wt.shape[0] < wt.shape[1] else csc_matrix(wt)
                                                        min_ = min(mat.data)
                                                        max_ = max(mat.data)
                                                        space = np.linspace(min_, max_, num=2 ** bits)
                                                        kmeans = KMeans(n_clusters=len(space), init=space.reshape(-1, 1),
                                                                        n_init=1,
                                                                        precompute_distances=True, algorithm="full")
                                                        kmeans.fit(mat.data.reshape(-1, 1))
                                                        new_weight = kmeans.cluster_centers_[kmeans.labels_].reshape(-1)
                                                        mat.data = new_weight
                                                        result[i][j] = mat.toarray()

                                                layer.weight.data = torch.from_numpy(result).to(dev)