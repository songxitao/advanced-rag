import os
import sys
import subprocess

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

STAGES = {
    "1": ("Stage 1: tests/evaluation_set_generator.py", "tests/evaluation_set_generator.py"),
    "2": ("Stage 2: tests/run_retrieval.py", "tests/run_retrieval.py"),
    "3": ("Stage 3: tests/generate_answers.py", "tests/generate_answers.py"),
    "4": ("Stage 4: tests/evaluate_results.py", "tests/evaluate_results.py")
}

def run_script(script_path, extra_args=None):
    if extra_args is None:
        extra_args = []
    print(f"\n>>> Running {script_path}...")
    abs_script_path = os.path.join(PROJECT_ROOT, script_path)
    try:
        # Use sys.executable to ensure we run in the same virtual environment
        env = os.environ.copy()
        env["PYTHONPATH"] = PROJECT_ROOT
        cmd = [sys.executable, abs_script_path] + extra_args
        result = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env)
        if result.returncode == 0:
            print(f"\n>>> {script_path} finished successfully.")
            return True
        else:
            print(f"\n>>> Error: {script_path} failed with exit code {result.returncode}.")
            return False
    except Exception as e:
        print(f"\n>>> Unexpected error running {script_path}: {e}")
        return False

def run_stage(key, extra_args=None):
    if key not in STAGES:
        return False
    name, path = STAGES[key]
    abs_path = os.path.join(PROJECT_ROOT, path)
    if not os.path.exists(abs_path):
        print(f"\n>>> Error: Script not found at {abs_path}")
        return False
    return run_script(path, extra_args)

def show_menu():
    print("\n" + "="*50)
    print("      RAG Evaluation Pipeline Control Manager")
    print("="*50)
    for key, (name, _) in STAGES.items():
        print(f" [{key}] {name}")
    print(" [A] Run all stages in sequence")
    print(" [Q] Quit")
    print("="*50)

def main():
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

    sanguo_mode = '--sanguo' in sys.argv
    extra_args = ['--sanguo'] if sanguo_mode else []
    if sanguo_mode:
        STAGES["1"] = ("Stage 1: tests/evaluation_set_generator_graph.py", "tests/evaluation_set_generator_graph.py")

    # Support non-interactive mode
    if '--all' in sys.argv:
        print("\n>>> [Non-Interactive] Running all stages in sequence...")
        
        # 清理旧的中间文件以保证重新运行的纯洁性
        old_files = [
            "tests/temp_data/test_sanguo_dataset.json",
            "tests/temp_data/retrieval_sanguo_results.json",
            "tests/temp_data/answer_sanguo_results.json",
            "tests/temp_data/evaluation_sanguo_scores.json"
        ]
        for f in old_files:
            abs_f = os.path.join(PROJECT_ROOT, f)
            if os.path.exists(abs_f):
                try:
                    os.remove(abs_f)
                    print(f"Removed old file: {f}")
                except Exception as e:
                    print(f"Warning: could not remove {f}: {e}")
                    
        all_success = True
        for key in STAGES.keys():
            if not run_stage(key, extra_args):
                print(f"\n>>> Pipeline stopped: Stage {key} failed or missing.")
                all_success = False
                break
        if all_success:
            print("\n>>> All stages executed successfully.")
        sys.exit(0 if all_success else 1)

    while True:
        show_menu()
        try:
            choice = input("Enter choice: ").strip().upper()
        except KeyboardInterrupt:
            print("\nExiting Pipeline Control Manager. Goodbye!")
            break
        except EOFError:
            print("\nEOF received. Exiting Pipeline Control Manager.")
            break

        if choice == 'Q':
            print("Exiting Pipeline Control Manager. Goodbye!")
            break
        elif choice == 'A':
            print("\n>>> Running all stages in sequence...")
            all_success = True
            for key in STAGES.keys():
                if not run_stage(key, extra_args):
                    print(f"\n>>> Pipeline stopped: Stage {key} failed or missing.")
                    all_success = False
                    break
            if all_success:
                print("\n>>> All stages executed successfully.")
        elif choice in STAGES:
            run_stage(choice, extra_args)
        else:
            print("Invalid choice, please try again.")

if __name__ == "__main__":
    main()
