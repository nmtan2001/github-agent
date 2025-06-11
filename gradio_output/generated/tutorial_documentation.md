# Python SDK Project Tutorial

Welcome to the Python SDK project tutorial! This guide will walk you through setting up and running a Python SDK project with various server and client examples. This tutorial is designed to be beginner-friendly and comprehensive, providing you with a solid foundation to explore more advanced features.

## Learning Objectives

By the end of this tutorial, you will be able to:

1. Understand the structure and purpose of a Python SDK project.
2. Set up the project environment and dependencies.
3. Run various server and client examples included in the project.
4. Troubleshoot common issues.
5. Explore advanced usage and customization.

## Prerequisites and Setup

Before you begin, ensure you have the following:

- **Python 3.7 or later**: Make sure Python is installed on your system. You can download it from the [official Python website](https://www.python.org/downloads/).
- **Git**: You will need Git to clone the project repository. Download it from the [official Git website](https://git-scm.com/).
- **A code editor**: Use any code editor of your choice, such as VSCode, PyCharm, or Sublime Text.

### Setup Instructions

1. **Clone the Repository**

   Open your terminal or command prompt and run the following command to clone the project repository:

   ```bash
   git clone https://github.com/your-username/python-sdk.git
   cd python-sdk
   ```

2. **Create a Virtual Environment**

   It's a good practice to use a virtual environment for Python projects to manage dependencies. Run the following commands:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. **Install Dependencies**

   Install the necessary dependencies using pip:

   ```bash
   pip install -r requirements.txt
   ```

## Step-by-Step Instructions

### 1. Running the Simple Auth Client

Navigate to the `examples/clients/simple-auth-client/mcp_simple_auth_client` directory and run the client:

```bash
cd examples/clients/simple-auth-client/mcp_simple_auth_client
python main.py
```

**Expected Output**: You should see logs indicating that the client has started and is attempting to authenticate.

### 2. Running the Simple Auth Server

Navigate to the `examples/servers/simple-auth/mcp_simple_auth` directory and run the server:

```bash
cd examples/servers/simple-auth/mcp_simple_auth
python server.py
```

**Expected Output**: The server should start and listen for incoming authentication requests.

### 3. Running Other Server Examples

You can explore other server examples by navigating to their respective directories and running the `server.py` files. For example, to run the simple prompt server:

```bash
cd examples/servers/simple-prompt/mcp_simple_prompt
python server.py
```

### 4. Running the CLI

The project includes a command-line interface (CLI) tool. You can run it as follows:

```bash
cd src/mcp/cli
python cli.py
```

**Expected Output**: The CLI should display available commands and options.

## Common Troubleshooting Tips

- **Module Not Found Error**: Ensure your virtual environment is activated and all dependencies are installed.
- **Port Already in Use**: If a server fails to start due to a port conflict, change the port number in the server script.
- **Permission Denied**: Ensure you have the necessary permissions to execute scripts and access network resources.

## Next Steps for Advanced Usage

Once you're comfortable with the basics, consider exploring the following:

- **Concurrency Testing**: Run the `tests/issues/test_188_concurrency.py` script to understand concurrency handling.
- **Customizing Servers**: Modify server scripts to add custom features or integrate with other services.
- **Extending the CLI**: Add new commands to the CLI tool to automate tasks or interact with your servers.

Congratulations on completing this tutorial! You now have a foundational understanding of the Python SDK project and can explore more advanced features and customizations. Happy coding!