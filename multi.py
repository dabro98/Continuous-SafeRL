import subprocess

# List of arguments to pass to main.py
envs = ["configs/t1/config_0.json", "configs/t1/config_1.json",
        "configs/t1/config_2.json", "configs/t1/config_3.json",
        "configs/t2/config_0.json", "configs/t2/config_1.json",
        "configs/t2/config_2.json", "configs/t2/config_3.json",
        "configs/t3/config_0.json", "configs/t3/config_1.json",
        "configs/t3/config_2.json", "configs/t3/config_3.json"]

# envs = {"configs/t4/config_0.json", "configs/t4/config_1.json",
#         "configs/t4/config_2.json", "configs/t4/config_3.json",
#         "configs/t5/config_0.json", "configs/t5/config_1.json",
#         "configs/t5/config_2.json", "configs/t5/config_3.json"}



# Path to the main.py script
main_script_path = "multi_main.py"

# Iterate through the arguments array and call main.py with each set of arguments
for env in envs:
    command = ["python", main_script_path] + [env]
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error calling main.py with arguments {env}: {e}")

