import argparse

from sys import exit as sys_exit
from torch import tensor as ttensor
from torch import get_num_threads
from pandas import read_csv, DataFrame
from pathlib import Path
from json import dump as json_dump
from signal import signal, SIGINT

from MOBSTERm import *


def signal2_handler(signum, frame):
    print("\nSignal 2 (Ctrl+C) detected. Exiting program now.")
    sys_exit(0)

signal(SIGINT, signal2_handler)

def uint_list(values_str: str) -> list[int]:
    values = []
    for value_str in values_str.split(','):
        try:
            value = int(value_str)
        except ValueError:
            raise ValueError('Error: expected an unsigned integer '
                             + f'value. Got "{value_str}".')
        if value < 0:
            raise ValueError('Error: expected an unsigned integer '
                             + f'value. Got "{value_str}".')
        values.append(value)

    return values

def fp_list(values: str) -> list[float]:
    try:
        return [float(x) for x in values.split(',')]
    except ValueError:
        raise ValueError('Error: expected floating value '
                         + 'list "<float>[,<float>]*"')

def karyotype(k_str: str) -> list[str]:
    kr = k_str.split(':')
    if len(kr) != 2:
        raise ValueError(f"\"{k_str}\" is not a karyotype")
    try:
        kr = [int(value) for value in kr]
    except ValueError:
        raise ValueError(f"\"{k_str}\" is not a karyotype")

    for k in kr:
        if k < 0:
            raise ValueError(f"\"{k_str}\" is not a karyotype")

    return kr

def kr_list(values: str) -> list[list[str]]:
    try:
        return [karyotype(x) for x in values.split(',')]
    except:
        raise ValueError('Error: expected karyotype '
                         + 'list "<karyotype>[,<karyotype>]*"')
    
def sample_list(samples_str: str) -> list[str]:
    return samples_str.split(',')

def validate_num_of_threads(value: str) -> int:
    try:
        ivalue = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"'{value}' is not a valid integer.")
        
    if ivalue <= 0:
        raise argparse.ArgumentTypeError("The number of threads must be " +
                                         f"greater than 0 (got {ivalue}).")

    return ivalue

def extract_sample_names(mutation_df: DataFrame,
                         samples: list[str]) -> list[str]:

    NV_columns = [col_name for col_name in mutation_df.columns
                    if col_name.endswith('.NV')]
    DP_columns = [col_name for col_name in mutation_df.columns
                    if col_name.endswith('.DP')]

    NV_sample_names = set([col_name[:-3] for col_name in NV_columns])
    DP_sample_names = set([col_name[:-3] for col_name in DP_columns])

    if samples is None:
        missing_NV = DP_sample_names - NV_sample_names
        if len(missing_NV)>0:
            raise ValueError(f"Missing NV for {missing_NV}")
        missing_DP = NV_sample_names - DP_sample_names
        if len(missing_DP)>0:
            raise ValueError(f"Missing DP for {missing_DP}")

        samples = list(DP_sample_names)
    else:
        sample_set = set(samples)
        missing_NV = sample_set - NV_sample_names
        if len(missing_NV)>0:
            raise ValueError(f"Missing NV for {missing_NV}")
        missing_DP = sample_set - DP_sample_names
        if len(missing_DP)>0:
            raise ValueError(f"Missing DP for {missing_DP}")

    return list(set(samples))

def save_images(mb: dict, image_dir: str, image_type: str,
                image_dpi: int) -> None:
    image_dir = Path(image_dir)

    if not image_dir.is_dir():
        image_dir.mkdir(parents=True, exist_ok=True)

    image_functions = {
        'plot_deltas': plot_deltas,
        'plot_responsib': plot_responsib,
        'plot_paretos': plot_paretos,
        'plot_betas': plot_betas,
        'plot_cluster_marginals': plot_cluster_marginals,
        'plot_mixing_proportions': plot_mixing_proportions,
        'plot_marginals_inference': plot_marginals_inference,
        'plot_scatter_inference': plot_scatter_inference
    }

    for image_funct_name, image_funct in image_functions.items():
        try:
            fig = image_funct(mb)

            filename = f"{image_funct_name}.{image_type}"
            fig.savefig(image_dir / filename, dpi=image_dpi)
        except Exception as ex:
            print(f'{filename} not produced because {str(ex)}.')

class PlotImageAction(argparse._StoreAction):
    def __call__(self, parser, namespace, values, option_string=None):
        super().__call__(parser, namespace, values, option_string)

        setattr(namespace, f"plot_images", True)

class AppManager(object):

    def _set_up_parser(self):
        num_threads = max(1, get_num_threads()-2)

        description = ('A tool to perform multivariate subclonal'
                       + ' deconvolution')
        epilog = ('See https://github.com/caravagnalab/MOBSTERm for usage'
                  + ' examples')
        parser = argparse.ArgumentParser(prog='MOBSTERm',
                                         description=description,
                                         epilog=epilog)

        parser.add_argument('data_filename',
                            help='The CSV mutation data filename')
        parser.add_argument('cluster_output_filename',
                            help='The CSV of the cluster output filename')
        parser.add_argument('-f', '--force-overwrite', dest='force_overwrite',
                            action='store_true',
                            help='Force output file overwrite')
        parser.add_argument('-i', '--num-iterations', dest='num_iter',
                            type=int, default=2000,
                            help='Number of SVI iterations for model fitting')
        parser.add_argument('-c', '--cluster-list', dest='clusters',
                            type=uint_list, default=list(range(2,5)),
                            help=('List of cluster numbers to consider'
                                  + ' (default: "2,3,4")'))
        parser.add_argument('-s', '--samples', dest='samples',
                            type=sample_list, default=None,
                            help='Names of the samples to be processed.')
        parser.add_argument('-p', '--purity', dest='purity', type=fp_list,
                            default=None,
                            help=('Purity of the sample(s)'
                                  + ' (default: "1,1,...")'))
        parser.add_argument('-k', '--karyotypes', dest='karyotypes',
                            type=kr_list, default=None,
                            help=('Karyotype of the sample(s)'
                                  + '  (default: "1:1,1:1,...")'))
        parser.add_argument('-S', '--seed-list', dest='seeds', type=uint_list,
                            default=[1,2],
                            help=('List of random seeds for reproducibility '
                                  + '(default: "1,2")'))
        parser.add_argument('-t', '--par-threshold', dest='par_threshold',
                            type=float, default=0.005,
                            help='Tolerance for parameter convergence')
        parser.add_argument('-l', '--loss-threshold', dest='loss_threshold',
                            type=float, default=0.01,
                            help='Tolerance for loss convergence')
        parser.add_argument('-r', '--learning-rate', dest='learning_rate',
                            type=float, default=0.01,
                            help='Learning rate for optimization')
        parser.add_argument('-L', '--log-filename', dest='log_filename',
                            type=str, default="",
                            help='The log output filename')
        parser.add_argument('-P', '--generate-plots', dest='generate_plots',
                            help='Generate plot images',
                            action=PlotImageAction)
        parser.add_argument('-d', '--dpi', dest='dpi',
                            type=float, default=300,
                            help='Image DPI (default: 300)',
                            action=PlotImageAction)
        parser.add_argument('-D', '--image-dir', dest='image_dir',
                            type=str, default='.',
                            help='Image directory (default: .)',
                            action=PlotImageAction)
        parser.add_argument('-T', '--image-type', dest='image_type',
                            type=str, default="png",
                            help='Image type (default: png)',
                            action=PlotImageAction)
        parser.add_argument('-n', '--num-of-threads', dest='num_of_threads',
                            default=num_threads, type=validate_num_of_threads,
                            help=f'Number of threads (default: {num_threads})')
        parser.add_argument('-q', '--quiet', dest='quiet',
                            action="store_true",
                            help='Avoid the progress bars')
        
        self.parser = parser

    def __init__(self):
        self._set_up_parser()

    def _collect_data(self):
        args = self.parser.parse_args()

        self.quiet = args.quiet

        if (Path(args.cluster_output_filename).is_file() and
                not args.force_overwrite):
            msg = (f'The file "{args.cluster_output_filename}" '
                   + 'already exists. Use the option "-f" '
                   + 'to overwrite it.')
            self.parser.error(msg)

        mutation_df = read_csv(args.data_filename)

        if 'mutation_id' not in mutation_df:
            try:
                mutation_df['mutation_id'] = build_mutation_ids(mutation_df)
            except ValueError as error:
                self.parser.error(error)

        samples = extract_sample_names(mutation_df, args.samples)

        if (args.purity is not None
                and len(samples) != len(args.purity)):
            raise ValueError("The sample name and purity lists "
                             + f"({samples} and {args.purity}, "
                             + "respectively) differ in size.")

        if (args.karyotypes is not None
                and len(samples) != len(args.karyotypes)):
            raise ("The sample name and karyotype lists "
                   + f"({samples} and {args.karyotypes}, "
                   + "respectively) differ in size.")

        NV_columns = [col + '.NV' for col in samples]
        NV_df = mutation_df[NV_columns]
        NV = ttensor(NV_df.values)

        DP_columns = [col + '.DP' for col in samples]
        DP_df = mutation_df[DP_columns]
        DP = ttensor(DP_df.values)

        if 'mutation_id' in mutation_df:
            mut_id = mutation_df['mutation_id']
        else:
            mut_id = build_mutation_ids(mutation_df)

        return NV, DP, mut_id.tolist(), samples, args
    
    def _save_log_data(best_fit, log_filename):
        log_data = {
            'step': list(range(1,len(best_fit['likelihood_per_step'])+1))
        }

        for key, value in best_fit['gradient_norms'].items():
            col_name = key.removesuffix("_param")
            log_data[col_name] = value

        log_data['likelihood'] = best_fit['likelihood_per_step']
        log_data['loss'] = best_fit['loss_per_step']

        log_df = pd.DataFrame(log_data)
        log_df.to_csv(log_filename, index=False)

    def _save_output(best_fit, output_filename):
        mutation_data = {
            'mutation_id': best_fit['mutation_id'],
            'cluster_id': best_fit['cluster_id']
        }

        cluster_df = pd.DataFrame(mutation_data)
        cluster_df.to_csv(output_filename, index=False)

    def run(self):
        NV, DP, mut_id, samples, args = self._collect_data()

        try:
            mb = model_mobster.fit(NV = NV, DP = DP, mut_id=mut_id,
                                   num_iter=args.num_iter, K=args.clusters,
                                   sample_names=samples, purity=args.purity,
                                   kr=args.karyotypes, seed_list=args.seeds,
                                   par_threshold=args.par_threshold,
                                   loss_threshold=args.loss_threshold,
                                   lr=args.learning_rate,
                                   num_of_threads=args.num_of_threads,
                                   quiet=self.quiet)
        except Exception as e:
            print("Error: " + str(e))
            sys_exit(0)

        best_fit = mb['best_fit']

        if not self.quiet:
            print()
            print("Results")

            output_data_names = ['bic', 'icl', 'final_likelihood', 'final_loss',
                                'n_components', 'used_components', 'seed']

            for data_name in output_data_names:
                print(f'{data_name}\t{best_fit[data_name]}')

        AppManager._save_output(best_fit, args.cluster_output_filename)
        if args.log_filename != "":
            AppManager._save_log_data(best_fit, args.log_filename)

        if getattr(args, 'plot_image', True):
            save_images(mb, args.image_dir, args.image_type, args.dpi)

def main():
    app_manager = AppManager()

    app_manager.run()

if __name__ == "__main__":
    main()
