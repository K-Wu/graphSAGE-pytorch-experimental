import sys
import os

from collections import defaultdict
import numpy as np


class DataCenter(object):
    """docstring for DataCenter"""

    def __init__(self, config, graphiler_loader_path):
        super(DataCenter, self).__init__()
        self.config = config
        self.graphiler_loader = self.import_from_path(
            "graphiler_loader", graphiler_loader_path
        )

    def parse_feat_dim_and_num_classes(self, config):
        feat_dim_raw = config["setting.feat_dim"]
        num_classes_raw = config["setting.num_classes"]
        try:
            feat_dim = int(feat_dim_raw)
        except ValueError as e:
            feat_dim = None
        try:
            num_classes = int(num_classes_raw)
        except ValueError as e:
            num_classes = None
        return feat_dim, num_classes

    def import_from_path(self, name, path):
        import importlib.util

        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        # uncomment if there is error from https://stackoverflow.com/a/67692
        # sys.modules[name] = module
        spec.loader.exec_module(module)
        return module

    def load_graphiler_dataSet(self, dataSet):

        g = self.graphiler_loader.graphiler_load_data(dataSet)

        # NB: KWU: use feat_dim and num_classes to generate random features and labels
        # NB: KWU: these two parameters should override dataset properties
        feat_dim, num_classes = self.parse_feat_dim_and_num_classes(self.config)

        if not feat_dim is None:
            import torch

            feat_data = torch.rand([g.number_of_nodes(), feat_dim])

        if not num_classes is None:
            import torch

            labels = torch.randint(num_classes, (g.number_of_nodes(),))

        # NB: KWU: create adj_lists from graph
        adj_lists = defaultdict(set)
        for edge in g.edges():
            adj_lists[edge[0]].add(edge[1])
            adj_lists[edge[1]].add(edge[0])

        setattr(self, dataSet + "_feats", feat_data)
        setattr(self, dataSet + "_labels", labels)

        test_indexs, val_indexs, train_indexs = self._split_data(feat_data.shape[0])

        setattr(self, dataSet + "_test", test_indexs)
        setattr(self, dataSet + "_val", val_indexs)
        setattr(self, dataSet + "_train", train_indexs)

        setattr(self, dataSet + "_feats", feat_data)
        setattr(self, dataSet + "_labels", labels)
        setattr(self, dataSet + "_adj_lists", adj_lists)
        raise NotImplementedError

    def load_dataSet(self, dataSet="cora"):
        if dataSet == "cora":
            cora_content_file = self.config["file_path.cora_content"]
            cora_cite_file = self.config["file_path.cora_cite"]

            feat_data = []
            labels = []  # label sequence of node
            node_map = {}  # map node to Node_ID
            label_map = {}  # map label to Label_ID
            with open(cora_content_file) as fp:
                for i, line in enumerate(fp):
                    info = line.strip().split()
                    feat_data.append([float(x) for x in info[1:-1]])
                    node_map[info[0]] = i
                    if not info[-1] in label_map:
                        label_map[info[-1]] = len(label_map)
                    labels.append(label_map[info[-1]])
            feat_data = np.asarray(feat_data)
            labels = np.asarray(labels, dtype=np.int64)

            # NB: KWU: generate artificial feat_data and labels here to override the original one if dictated by the configurations file
            feat_dim, num_classes = self.parse_feat_dim_and_num_classes(self.config)

            if not feat_dim is None:
                import torch

                feat_data = torch.rand([g.number_of_nodes(), feat_dim])

            if not num_classes is None:
                import torch

                labels = torch.randint(num_classes, (g.number_of_nodes(),))

            adj_lists = defaultdict(set)
            with open(cora_cite_file) as fp:
                for i, line in enumerate(fp):
                    info = line.strip().split()
                    assert len(info) == 2
                    paper1 = node_map[info[0]]
                    paper2 = node_map[info[1]]
                    adj_lists[paper1].add(paper2)
                    adj_lists[paper2].add(paper1)

            assert len(feat_data) == len(labels) == len(adj_lists)
            test_indexs, val_indexs, train_indexs = self._split_data(feat_data.shape[0])

            setattr(self, dataSet + "_test", test_indexs)
            setattr(self, dataSet + "_val", val_indexs)
            setattr(self, dataSet + "_train", train_indexs)

            setattr(self, dataSet + "_feats", feat_data)
            setattr(self, dataSet + "_labels", labels)
            setattr(self, dataSet + "_adj_lists", adj_lists)

        elif dataSet == "pubmed":
            pubmed_content_file = self.config["file_path.pubmed_paper"]
            pubmed_cite_file = self.config["file_path.pubmed_cites"]

            feat_data = []
            labels = []  # label sequence of node
            node_map = {}  # map node to Node_ID
            with open(pubmed_content_file) as fp:
                fp.readline()
                feat_map = {
                    entry.split(":")[1]: i - 1
                    for i, entry in enumerate(fp.readline().split("\t"))
                }
                for i, line in enumerate(fp):
                    info = line.split("\t")
                    node_map[info[0]] = i
                    labels.append(int(info[1].split("=")[1]) - 1)
                    tmp_list = np.zeros(len(feat_map) - 2)
                    for word_info in info[2:-1]:
                        word_info = word_info.split("=")
                        tmp_list[feat_map[word_info[0]]] = float(word_info[1])
                    feat_data.append(tmp_list)

            feat_data = np.asarray(feat_data)
            labels = np.asarray(labels, dtype=np.int64)

            adj_lists = defaultdict(set)
            with open(pubmed_cite_file) as fp:
                fp.readline()
                fp.readline()
                for line in fp:
                    info = line.strip().split("\t")
                    paper1 = node_map[info[1].split(":")[1]]
                    paper2 = node_map[info[-1].split(":")[1]]
                    adj_lists[paper1].add(paper2)
                    adj_lists[paper2].add(paper1)

            assert len(feat_data) == len(labels) == len(adj_lists)
            test_indexs, val_indexs, train_indexs = self._split_data(feat_data.shape[0])

            setattr(self, dataSet + "_test", test_indexs)
            setattr(self, dataSet + "_val", val_indexs)
            setattr(self, dataSet + "_train", train_indexs)

            setattr(self, dataSet + "_feats", feat_data)
            setattr(self, dataSet + "_labels", labels)
            setattr(self, dataSet + "_adj_lists", adj_lists)

        else:
            # use graphiler data loader
            self.load_graphiler_dataSet(dataSet)

    def _split_data(self, num_nodes, test_split=3, val_split=6):
        rand_indices = np.random.permutation(num_nodes)

        test_size = num_nodes // test_split
        val_size = num_nodes // val_split
        train_size = num_nodes - (test_size + val_size)

        test_indexs = rand_indices[:test_size]
        val_indexs = rand_indices[test_size : (test_size + val_size)]
        train_indexs = rand_indices[(test_size + val_size) :]

        return test_indexs, val_indexs, train_indexs
