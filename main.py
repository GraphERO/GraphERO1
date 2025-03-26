import time
import arg
import os
os.environ['MPLCONFIGDIR'] = os.getcwd() + "/configs/"

def main():
    parser = arg.get_parser()
    args = parser.parse_args()

    if args.embedder == 'graphero':
        from models.graphero import graphero
        embedder = graphero(args)
        
    t_total = time.time()
    embedder.training()
    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

if __name__ == '__main__':
    main()
