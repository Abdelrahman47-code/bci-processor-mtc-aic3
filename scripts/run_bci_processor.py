# scripts/run_bci_processor.py
import argparse
from bci_processor.processor import EnhancedBCIProcessor
from bci_processor.utils import setup_logging

def main():
    parser = argparse.ArgumentParser(description="Run BCI Processor for EEG signal classification.")
    parser.add_argument('--config', type=str, default='bci_processor/config.yaml', help='Path to configuration file')
    parser.add_argument('--output', type=str, default='enhanced_submission.csv', help='Path to output submission file')
    parser.add_argument('--log-level', type=str, default='INFO', help='Logging level (DEBUG, INFO, WARNING, ERROR)')
    args = parser.parse_args()
    
    setup_logging(args.log_level)
    processor = EnhancedBCIProcessor(args.config)
    processor.train()
    submission = processor.generate_predictions(args.output)
    print("\nSubmission preview:")
    print(submission.head(10))
    print("\nPrediction distribution:")
    print(submission['label'].value_counts())

if __name__ == "__main__":
    main()