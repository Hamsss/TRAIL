import argparse

class Setting():
    
    def __init__(self):
        pass
    
    def init_state(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--cuda', type=int, default = 0,
                            help= 'cuda number')        
        parser.add_argument('--epochs', type=int, default=1000,
                            help='Number of epochs to train.')
        parser.add_argument('--model', type=str, default = 'GCN',
                            help= 'graph model GCN/GraphSage/SGC')
        parser.add_argument('--dataset', type=str, default = 'Cora',
                            help= 'set the dataset')        
        parser.add_argument('--seed', type=float, default = 42,
                            help= 'set the seed')
        parser.add_argument('--layer', type=int, default = 128,
                            help= 'Number of hidden layer')
        # Hyperparameters
        parser.add_argument('--lr', type=float, default=1e-2,
                            help='Initial learning rate.')
        parser.add_argument('--weight_decay', type=float, default=5e-4,
                            help='Weight decay (L2 loss on parameters).')
        parser.add_argument('--dropout', type=float, default=0.6,
                            help='Dropout rate (1 - keep probability).')
        parser.add_argument('--hidden', type=int, default=128,
                            help='Number of hidden layer dimension.')
        parser.add_argument('--momentum', type=float, default=0.05,
                            help='batch norm momentum.')        
        args = parser.parse_args()
        return args

'''
available datasets
Cora, Citeseer, Pubmed, CS, Physics, Computers, Photo, Wiki, Cornell, Texas, 
Wisconsin, Chameleon, Squirrel, Actor, DBLP, CoraFull, Facebook

Below number of training set and test set are following ratio of 6:4
Cora - total_data: 2708 (trainset: 2166 test: 542), feature: 1433, edge: 10556, class: 7
Citeseer - total_data: 3327 (trainset: 2662 test: 665), feature: 3703, edge: 9104, class: 6
Pubmed - total_data: 19717 (trainset: 15774 test: 3943), feature: 500, edge: 88648, class: 3
CS - total_data: 18333 (trainset: 14666 test: 3667), feature: 6805, edge: 163788, class: 15
Physics - total_data: 34493 (trainset: 27594 test: 6899), feature: 8415, edge: 495924, class: 5
Computers - total_data: 13752 (trainset: 11002 test: 2750), feature: 767, edge: 491722, class: 10
Photo - total_data: 7650 (trainset: 6120 test: 1530), feature: 745, edge: 238162, class: 8
Wiki - total_data: 2405 (trainset: 1924 test: 481), feature: 4973, edge: 17981, class: 17
Cornell - total_data: 183 (trainset: 146 test: 37), feature: 1703, edge: 298, class: 5
Texas - total_data: 183 (trainset: 146 test: 37), feature: 1703, edge: 325, class: 5
Wisconsin - total_data: 251 (trainset: 201 test: 50), feature: 1703, edge: 515, class: 5
Chameleon - total_data: 2277 (trainset: 1822 test: 455), feature: 2325, edge: 36101, class: 5
Squirrel - total_data: 5201 (trainset: 4161 test: 1040), feature: 2089, edge: 217073, class: 5
Actor - total_data: 7600 (trainset: 6080 test: 1520), feature: 932, edge: 30019, class: 5
DBLP - total_data: 17716 (trainset: 10630 test: 7086), feature: 1639 edge: 105734, label: 4
CoraFull - total_data: 19793 (trainset: 11876 test: 7917), feature: 8710 edge: 126842, label: 70
Facebook - total_data: 22470 (trainset: 13482 test: 8988), feature: 128 edge: 342004, label: 4

seed
100, 200 ... 1000
'''