import argparse


def get_settings():
    parser = argparse.ArgumentParser()
    # Seeds for random sampling
    parser.add_argument('--seed', default=1)
    parser.add_argument('--seed_data', default=1)
    parser.add_argument('--batch_size', default=25)
    parser.add_argument('--num_epochs', default=400)
    parser.add_argument('--learning_rate', type=float, default=0.0003)
    # It is possible use pretrained model for fine tuning
    parser.add_argument('--use_pretrained', default=False)
    # Number of binary features to be used for embeddings
    parser.add_argument('--num_features', default=256)
    # parser.add_argument('--num_features', default=16) - for cifar data
    # Beta parameter
    parser.add_argument('--beta', type=float, default=2.0)
    # Gamma parameter parameter
    parser.add_argument('--gamma', type=float, default=0.001)
    # Lambda BRE
    parser.add_argument('--lambda_bre', type=float, default=0.01)
    #parser.add_argument('--lambda_bre', type=float, default=0.001) - for cifar_data
    # Lambda DMR
    parser.add_argument('--lambda_dmr', type=float, default=0.05)
    #Location of pretrained weights
    parser.add_argument('--generator_pretrained', default='generator_pretrained.npz')
    parser.add_argument('--discriminator_pretrained', default='discriminator_pretrained.npz')

    # Where weight for model should be saved
    parser.add_argument('--generator_out', default='/home/mzieba/workspace_docker/models/generator_brown.npz')
    parser.add_argument('--discriminator_out', default='/home/mzieba/workspace_docker/models/discriminator_brown.npz')

    # Dir, where data should be downloaded
    parser.add_argument('--data_dir', type=str, default='/home/mzieba/workspace_docker/data_2/')

    parser.add_argument('--dataset_type', type=str, default='brown')
    #parser.add_argument('--dataset_type', type=str, default='cifar10')
    # Specify the train and test data for Brown Dataset
    parser.add_argument('--data_name', type=str, default='yosemiteL')
    parser.add_argument('--test_data', type=str, default='notredame')

    args = parser.parse_args()
    return args
