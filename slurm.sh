#!/bin/bash

# Parse arguments into associative array
declare -A args
while [[ $# -gt 0 ]]; do
    if [[ $1 == --* ]]; then
        arg=${1/--/}   # Remove prefix '--'
        if [[ $arg == *"="* ]]; then  # If contains equals sign
            key=${arg%%=*}  # Extract key before '='
            value=${arg#*=} # Extract value after '='
            args["$key"]=$value
        else
            # Handle flags without values
            key=$arg
            args["$key"]=""
        fi
    fi
    shift
done

# Construct the Python command
python_command="python -m ${args["module"]}"
for key in "${!args[@]}"; do
    if [[ $key != "module" ]]; then
        if [[ -z "${args[$key]}" ]]; then
            # If value is empty (flag), just add the flag
            python_command+=" --$key"
        else
            # Otherwise add key-value pair with proper quoting
            python_command+=" --$key='${args[$key]}'"
        fi
    fi
done

# Execute the Python command
echo $python_command
eval $python_command