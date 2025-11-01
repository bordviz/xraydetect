from utils import check_cuda
from scripts import SIXRayDatasetChecker

data_yaml_path = 'data/SIXray_mini/data.yaml'

def main():
    print(f"{'='*10} Check GPU avalible {'='*10}\n")
    check_cuda()

    print(f"{'='*10} Check dataset {'='*10}\n")
    checker = SIXRayDatasetChecker(data_yaml_path=data_yaml_path)
    checker.check_all_splits()

if __name__ == '__main__':
    main()