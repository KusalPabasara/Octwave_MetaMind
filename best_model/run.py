import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(ROOT, '..'))

ENV = {
    'DATA_DIR': os.environ.get('DATA_DIR', os.path.join(PROJECT_ROOT, 'data/data')),
    'TEST_CSV': os.environ.get('TEST_CSV', os.path.join(PROJECT_ROOT, 'test.csv')),
    'MODEL_PATH': os.environ.get('MODEL_PATH', os.path.join(PROJECT_ROOT, 'resnet50_f1_optimized.pth')),
    'LABEL_ENCODERS': os.environ.get('LABEL_ENCODERS', os.path.join(PROJECT_ROOT, 'label_encoders_resnet50.pth')),
    'SUBMISSION_PATH': os.environ.get('SUBMISSION_PATH', os.path.join(PROJECT_ROOT, 'submission_resnet50_HIGH_RISK_PLUS.csv')),
    'BATCH_SIZE': os.environ.get('BATCH_SIZE', '16'),
    'IMAGE_SIZE': os.environ.get('IMAGE_SIZE', '224'),
    'NUM_WORKERS': os.environ.get('NUM_WORKERS', '4'),
    'THRESH_ADDED': os.environ.get('THRESH_ADDED', '0.48'),
    'THRESH_REMOVED': os.environ.get('THRESH_REMOVED', '0.68'),
    'THRESH_CHANGED': os.environ.get('THRESH_CHANGED', '0.70'),
}

def main():
    env = os.environ.copy()
    env.update(ENV)

    python = sys.executable
    test_py = os.path.join(ROOT, 'test.py')

    print('='*70)
    print('Running best_so_far1.0/test.py with HIGH_RISK_PLUS thresholds')
    print('='*70)
    print('Submission path:', ENV['SUBMISSION_PATH'])

    proc = subprocess.run([python, test_py], env=env)
    sys.exit(proc.returncode)

if __name__ == '__main__':
    main()
