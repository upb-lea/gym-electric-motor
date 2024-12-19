import os
import subprocess

import pytest

file_names = ["classic_controllers_dc_motor_example.py", 
              "classic_controllers_ind_motor_example.py",
              "classic_controllers_synch_motor_example.py",
              "custom_classic_controllers_dc_motor_example.py" ,
              "custom_classic_controllers_ind_motor_example.py",
              "custom_classic_controllers_synch_motor_example.py",
              "integration_test_classic_controllers_dc_motor.py"
             ] 
     
@pytest.mark.parametrize("file_name", file_names)
def test_run_classic_controllers(file_name):
    # Run the script and capture the output
    directory = "examples" 
    subdirectory = "classic_controllers" 
    result = subprocess.run(["python", os.path.join(directory, subdirectory, file_name)], capture_output=True, text=True)

    # Check if the script ran successfully (exit code 0)
    assert result.returncode == 0, file_name + " did not exit successfully"


    
   
  