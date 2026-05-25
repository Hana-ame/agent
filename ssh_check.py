import subprocess

def check_ssh():
    print("Attempting to SSH to bwh.moonchan.xyz...")
    try:
        # -o BatchMode=yes: Disable password prompting
        # -o ConnectTimeout=5: Timeout after 5 seconds
        result = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", "bwh.moonchan.xyz"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            print("Successfully connected to bwh.moonchan.xyz!")
            print(result.stdout)
        else:
            print(f"Failed to connect. Return code: {result.returncode}")
            print(f"Error: {result.stderr}")
    except subprocess.TimeoutExpired:
        print("SSH connection timed out.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    check_ssh()
