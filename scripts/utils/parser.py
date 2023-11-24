from argparse import ArgumentParser

class CustomParser(ArgumentParser): 
    def __init__(self):
        # Parser
        super(CustomParser,self).__init__()
        self.add_argument('--env', type=str, required=True)
        self.add_argument('--dir', type=str, default='')
        self.add_argument('--tasks', type=str, nargs='*', default='all')