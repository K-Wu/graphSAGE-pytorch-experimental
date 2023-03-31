import sys
import os
import torch
import argparse
import pyhocon
import random

from src.dataCenter import *
from src.utils import *
from src.models import *

parser = argparse.ArgumentParser(description="pytorch version of GraphSAGE")

parser.add_argument("--dataSet", type=str, default="cora")
parser.add_argument("--agg_func", type=str, default="MEAN")
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument(
    "--b_sz", type=int, default=1000, help="batch size"
)  # KWU: set to 1000 this is the batch size
parser.add_argument("--seed", type=int, default=824)
parser.add_argument("--cuda", action="store_true", help="use CUDA")
parser.add_argument("--gcn", action="store_true")
parser.add_argument("--learn_method", type=str, default="sup")
parser.add_argument("--unsup_loss", type=str, default="normal")
parser.add_argument("--max_vali_f1", type=float, default=0)
parser.add_argument("--name", type=str, default="debug")
parser.add_argument("--config", type=str, default="./src/experiments.conf")
parser.add_argument("--unique_after_sample", action="store_true")
# parser.add_argument("--graphiler_loader_path", type=str, default="../dgl_prelim/examples/common_utils_prelim/graphiler_datasets.py")
parser.add_argument("--use_unified_tensor", action="store_true")
args = parser.parse_args()

if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        device_id = torch.cuda.current_device()
        print("using device", device_id, torch.cuda.get_device_name(device_id))

device = torch.device("cuda" if args.cuda else "cpu")
print("DEVICE:", device)

if __name__ == "__main__":
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # load config file
    config = pyhocon.ConfigFactory.parse_file(args.config)

    # load data
    ds = args.dataSet
    dataCenter = DataCenter(
        config, graphiler_loader_path=config["graphiler_loader.path"]
    )
    dataCenter.load_dataSet(ds)
    features = torch.FloatTensor(getattr(dataCenter, ds + "_feats")).to(device)

    if args.use_unified_tensor:
        graphSage = GraphSage(
            config["setting.num_layers"],
            config["setting.fan_out"],
            features.size(1),
            config["setting.hidden_emb_size"],
            None,  # pass feature tensor after unified
            getattr(dataCenter, ds + "_adj_lists"),
            device,
            use_unified_tensor=args.use_unified_tensor,
            unique_after_sample=args.unique_after_sample,
            gcn=args.gcn,
            agg_func=args.agg_func,
        )
        graphSage.to(device)
        features = features.to("unified")
        graphSage.raw_features = features

    else:
        graphSage = GraphSage(
            config["setting.num_layers"],
            config["setting.fan_out"],
            features.size(1),
            config["setting.hidden_emb_size"],
            features,
            getattr(dataCenter, ds + "_adj_lists"),
            device,
            use_unified_tensor=args.use_unified_tensor,
            unique_after_sample=args.unique_after_sample,
            gcn=args.gcn,
            agg_func=args.agg_func,
        )
        graphSage.to(device)

    num_labels = len(set(getattr(dataCenter, ds + "_labels")))
    classification = Classification(config["setting.hidden_emb_size"], num_labels)
    classification.to(device)

    unsupervised_loss = UnsupervisedLoss(
        getattr(dataCenter, ds + "_adj_lists"),
        getattr(dataCenter, ds + "_train"),
        device,
    )

    if args.learn_method == "sup":
        print("GraphSage with Supervised Learning")
    elif args.learn_method == "plus_unsup":
        print("GraphSage with Supervised Learning plus Net Unsupervised Learning")
    else:
        print("GraphSage with Net Unsupervised Learning")

    # TODO: KWU: measure time for each epoch and/or step
    for epoch in range(args.epochs):
        print("----------------------EPOCH %d-----------------------" % epoch)
        graphSage, classification = apply_model(
            dataCenter,
            ds,
            graphSage,
            classification,
            unsupervised_loss,
            args.b_sz,
            args.unsup_loss,
            device,
            args.learn_method,
        )
        if (epoch + 1) % 2 == 0 and args.learn_method == "unsup":
            classification, args.max_vali_f1 = train_classification(
                dataCenter,
                graphSage,
                classification,
                ds,
                device,
                args.max_vali_f1,
                args.name,
                args.b_sz,
            )
        if args.learn_method != "unsup":
            args.max_vali_f1 = evaluate(
                dataCenter,
                ds,
                graphSage,
                classification,
                device,
                args.max_vali_f1,
                args.name,
                epoch,
            )
