import os
import sys
import subprocess
from datetime import datetime


def run_tests():
    # 获取当前目录下所有的Python文件
    test_files = [
        f
        for f in os.listdir(".")
        if f.endswith(".py") and f != os.path.basename(__file__)
    ]

    # 创建输出文件名,包含当前日期和时间
    output_file = f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

    with open(output_file, "w", encoding="utf-8") as out_file:
        for test_file in test_files:
            out_file.write(f"\n{'='*50}\n")
            out_file.write(f"Running test: {test_file}\n")
            out_file.write(f"{'='*50}\n\n")

            try:
                # 运行测试文件并捕获输出
                result = subprocess.run(
                    [sys.executable, test_file],
                    capture_output=True,
                    text=True,
                    timeout=300,
                )  # 5分钟超时

                # 写入标准输出
                out_file.write("Standard Output:\n")
                out_file.write(result.stdout)

                # 如果有错误输出,则写入错误信息
                if result.stderr:
                    out_file.write("\nError Output:\n")
                    out_file.write(result.stderr)

            except subprocess.TimeoutExpired:
                out_file.write(f"Error: Test {test_file} timed out after 5 minutes.\n")
            except Exception as e:
                out_file.write(f"Error running test {test_file}: {str(e)}\n")

            out_file.write("\n")  # 为下一个测试文件添加空行

    print(f"Test results have been saved to {output_file}")


if __name__ == "__main__":
    run_tests()
