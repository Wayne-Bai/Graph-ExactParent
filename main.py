from train import *

if __name__ == '__main__':
    # All necessary arguments are defined in args.py
    args = Args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda)
    print('CUDA', args.cuda)
    print('File name prefix', args.fname)
    # check if necessary directories exist
    if not os.path.isdir(args.model_save_path):
        os.makedirs(args.model_save_path)
    if not os.path.isdir(args.graph_save_path):
        os.makedirs(args.graph_save_path)
    if not os.path.isdir(args.figure_save_path):
        os.makedirs(args.figure_save_path)
    if not os.path.isdir(args.timing_save_path):
        os.makedirs(args.timing_save_path)
    if not os.path.isdir(args.figure_prediction_save_path):
        os.makedirs(args.figure_prediction_save_path)
    if not os.path.isdir(args.nll_save_path):
        os.makedirs(args.nll_save_path)

    time = strftime("%Y-%m-%d %H:%M:%S", gmtime())
    # logging.basicConfig(filename='logs/train' + time + '.log', level=logging.DEBUG)
    if args.clean_tensorboard:
        if os.path.isdir("tensorboard"):
            shutil.rmtree("tensorboard")
    configure("tensorboard/run" + time, flush_secs=5)

    graphs = create_graphs.create(args)

    # split datasets
    random.seed(123)
    shuffle(graphs)
    graphs_train = graphs
    graphs_len = len(graphs)
    # graphs_test = graphs[int(0.8 * graphs_len):]
    # graphs_train = graphs[0:int(0.8*graphs_len)]
    graphs_test = graphs_train

    args.max_num_node = max([graphs[i].number_of_nodes() for i in range(len(graphs))])
    max_num_edge = max([graphs[i].number_of_edges() for i in range(len(graphs))])
    min_num_edge = min([graphs[i].number_of_edges() for i in range(len(graphs))])

    # args.max_num_node = 2000
    # show graphs statistics
    print('total graph num: {}, training set: {}'.format(len(graphs), len(graphs_train)))
    print('max number node: {}'.format(args.max_num_node))
    print('max/min number edge: {}; {}'.format(max_num_edge, min_num_edge))
    print('max previous node: {}'.format(args.max_prev_node))

    # save ground truth graphs
    ## To get train and test set, after loading you need to manually slice
    save_graph_list(graphs, args.graph_save_path + args.fname_train + '0.dat')
    save_graph_list(graphs, args.graph_save_path + args.fname_test + '0.dat')
    print('train and test graphs saved at: ', args.graph_save_path + args.fname_test + '0.dat')

    ### dataset initialization

    train_set = Graph_sequence_sampler_pytorch(graphs_train, args, max_prev_node=args.max_prev_node,
                                                   max_num_node=args.max_num_node)
    test_set = Graph_sequence_sampler_pytorch(graphs_test, args, max_prev_node=args.max_prev_node,
                                                  max_num_node=args.max_num_node)
    sample_strategy = torch.utils.data.sampler.WeightedRandomSampler(
        [1.0 / len(train_set) for i in range(len(train_set))],
        num_samples=args.batch_size * args.batch_ratio, replacement=True)
    train_set_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, num_workers=args.num_workers,
                                                   sampler=sample_strategy)
    sample_strategy = torch.utils.data.sampler.WeightedRandomSampler(
        [1.0 / len(test_set) for i in range(len(test_set))],
        num_samples=args.test_total_size, replacement=True)
    test_set_loader = torch.utils.data.DataLoader(test_set, batch_size=args.test_batch_size,
                                                  num_workers=args.num_workers,
                                                  sampler=sample_strategy)

    ### model initialization

    rnn = GRU_plain(input_size=args.max_node_feature_num+args.max_child_node, embedding_size=args.embedding_size_rnn,
                        hidden_size=args.hidden_size_rnn, num_layers=args.num_layers, has_input=True,
                        has_output=True, output_size=args.hidden_size_rnn_output).cuda()
    node_f_gen = MLP_plain(h_size=args.hidden_size_rnn_output, embedding_size=args.embedding_size_output,
                               y_size=args.max_node_feature_num+args.max_child_node).cuda()
    output = GRU_plain(input_size=args.max_node_feature_num+args.max_child_node, embedding_size=args.embedding_size_rnn_output,
                           hidden_size=args.hidden_size_rnn_output, num_layers=args.num_layers, has_input=True,
                           has_output=True,
                           output_size=args.max_node_feature_num+args.max_child_node).cuda()  # TODO: understand input_size, output_size ?
    edge_f_gen = None

    ### start training
    train(args, train_set_loader, rnn, output, node_f_gen, edge_f_gen, test_set=test_set_loader)


