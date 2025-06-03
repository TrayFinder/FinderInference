
#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "Running pylint on src"
echo "---------------------------------"
pylint production/

echo -e "\n\n --------------------------------- \n\n"
echo "Running overall score calculation"
pylint production/ | grep rated
