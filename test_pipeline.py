import argparse

from engine.rule_fix import RuleBaseEngine
from utils.utils import save_csv


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parser for pipeline test")
    parser.add_argument("--config_path", type=str, required=True, help='config rule base')

    args = parser.parse_args()

    rule_base = RuleBaseEngine(config_path=args.config_path)
    result = rule_base()
#     save_csv(dictionary_result=result, path_save=rule_base.path_save_result)
