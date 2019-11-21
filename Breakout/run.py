from Breakout import Game
import DQN
import argparse

def run(model):
    pass

def train(model_name):
    pass

def create_arguments():
    parser = argparse.ArgumentParser(description="Arguments to train or run Breakout reinforcement project.  Append -h or --help for more detail.")

    parser.add_argument('--file','-f',action="store",dest="model",default="./model.h5",help="Pass the model you wish to run or save to.")
    parser.add_argument('--train','-t', action="store_true", dest="train",default=False, help="This argument will train the model.")
    parser.add_argument('--run','-r', action="store_true", dest="run",default=False, help="This argument will run the model.")
    return parser.parse_args()


if __name__ == "__main__":

    args = create_arguments()

    save_path = "./"
    environment = "BreakoutDeterministic-v4"

    game = Game()

    if args.train:
        train(args.model)

    if args.run:
        run(args.model)

