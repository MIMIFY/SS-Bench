#!/bin/bash

# 定义Python命令
PY_COMMAND="/data/anaconda3/envs/py310/bin/python github/hierarchical_instruct/bootstrap_stories_depth.py"

# 无限循环，直到用户手动停止脚本
while true; do
    # 运行Python命令
    echo "Starting Python command: $PY_COMMAND"
    eval $PY_COMMAND

    # 检查命令退出状态
    exit_status=$?

    # 如果命令成功执行（exit status 0），则退出循环
    if [ $exit_status -eq 0 ]; then
        echo "Python command completed successfully. Exiting loop."
        break
    else
        # 如果命令失败，等待一段时间（例如10秒）后重试
        echo "Python command failed with exit status $exit_status. Retrying in 10 seconds..."
        sleep 10
    fi
done


# 要使用这个脚本，请按照以下步骤操作：

# 1. 将上述代码保存到一个文件中，例如 `run_python_command.sh`。
# 2. 使脚本可执行：运行 `chmod +x run_python_command.sh`。
# 3. 运行脚本：执行 `./run_python_command.sh`。

# 脚本将会开始执行Python命令。如果命令因为任何原因中断或失败，脚本会在10秒后重试。如果命令成功完成（退出状态为0），脚本将打印一条消息并退出循环。

# 请注意，这个脚本假设您的Python命令可能会失败并需要重试。如果您的命令总是失败，脚本将不断重试。您可能需要检查命令和环境以确保它可以成功执行。此外，如果您希望在一定次数的失败尝试后停止重试，您可以修改脚本以包含一个计数器和最大尝试次数的限制。