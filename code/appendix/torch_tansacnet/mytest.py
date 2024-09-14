import unittest
import importlib
#import sys
import os

"""
MYTEST is a test suite that runs the test cases in TanSacNet package

This test script works with the unittest module.

Requirements:
    torch, torch_dct, parameterized

Copyright (c) 2024, Shogo MURAMATSU

All rights reserved.

Contact address: Shogo MURAMATSU,
    Faculty of Engineering, Niigata University,
    8050 2-no-cho Ikarashi, Nishi-ku,
    Niigata, 950-2181, JAPAN

https://www.eng.niigata-u.ac.jp/~msiplab/

"""

def suite(testclass):
    # TestLoader インスタンスを作成
    loader = unittest.TestLoader()

    # クラス内のすべてのテストケースをロードしてスイートに追加
    suite = unittest.TestSuite()
    suite.addTest(loader.loadTestsFromTestCase(testclass))

    return suite

if __name__ == "__main__":
    #args = sys.argv

    # 現在のディレクトリを取得
    current_dir = os.path.dirname(__file__)

    # ディレクトリ内のすべての test_*.py ファイルをリスト化
    test_modules = [f[:-3] for f in os.listdir(current_dir) if f.startswith("test_") and f.endswith(".py") ]

    # テストの一括実行
    for module_name in test_modules:
        try:
            # モジュールのインポート
            module = importlib.import_module(module_name)
            class_name = module_name[5].upper()+module_name[6:]+"TestCase"
            # クラスの取得
            test_class = getattr(module,class_name)
            # テストの実行
            print(class_name)
            runner = unittest.TextTestRunner()
            runner.run(suite(test_class))
        except ModuleNotFoundError:
            print(f"モジュール '{module_name}' が見つかりません。")
        except AttributeError:
            print(f"クラス '{class_name}' はモジュール '{module}' に存在しません。")
 