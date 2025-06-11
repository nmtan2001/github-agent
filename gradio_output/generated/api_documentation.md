# Python SDK API Documentation

## API Overview

The Python SDK provides a comprehensive set of tools and utilities for interacting with the MCP (Modular Communication Protocol) environment. It includes modules for handling authentication, session management, and HTTP communication, among others. This documentation aims to guide developers through the core modules, classes, and functions, providing detailed descriptions, usage examples, and error handling strategies.

## Core Modules

### Module: `types`

**Path**: `src/mcp/types.py`

This module defines various parameter types used throughout the SDK.

#### Classes

- **`RequestParams`**: Represents basic request parameters.
- **`PaginatedRequestParams`**: Represents parameters for paginated requests.
- **`NotificationParams`**: Represents parameters for notifications.

### Module: `claude`

**Path**: `src/mcp/cli/claude.py`

This module provides utilities for integrating with the Claude application.

#### Functions

- **`get_claude_config_path()`** -> `Path | None`
  - **Description**: Retrieves the Claude configuration directory based on the platform.
  - **Example**:
    ```python
    from mcp.cli.claude import get_claude_config_path

    config_path = get_claude_config_path()
    print(config_path)
    ```

- **`get_uv_path()`** -> `str`
  - **Description**: Retrieves the full path to the UV executable.
  - **Example**:
    ```python
    from mcp.cli.claude import get_uv_path

    uv_path = get_uv_path()
    print(uv_path)
    ```

- **`update_claude_config(file_spec, server_name)`** -> `bool`
  - **Description**: Adds or updates a FastMCP server in Claude's configuration.
  - **Parameters**:
    - `file_spec`: `Path` - Path to the server file.
    - `server_name`: `str` - Name of the server.
  - **Example**:
    ```python
    from mcp.cli.claude import update_claude_config

    success = update_claude_config('/path/to/server', 'MyServer')
    print(success)
    ```

### Module: `auth`

**Path**: `src/mcp/client/auth.py`

Implements OAuth2 authentication with PKCE and automatic token refresh.

#### Classes

- **`TokenStorage`**: Protocol for token storage implementations.
- **`OAuthClientProvider`**: Manages OAuth flow with automatic client registration and token handling.
  - **Methods**:
    - `__init__(self, server_url, client_metadata, storage, redirect_handler, callback_handler, timeout)`
    - `_generate_code_verifier(self) -> str`
    - `_generate_code_challenge(self, code_verifier) -> str`
    - `_get_authorization_base_url(self, server_url) -> str`
    - `_has_valid_token(self) -> bool`

#### Authentication Example

```python
from mcp.client.auth import OAuthClientProvider, TokenStorage

class MyTokenStorage(TokenStorage):
    # Implement token storage methods

provider = OAuthClientProvider(
    server_url='https://api.example.com',
    client_metadata={'client_id': 'my-client-id'},
    storage=MyTokenStorage(),
    redirect_handler=None,
    callback_handler=None,
    timeout=30
)

if provider._has_valid_token():
    print("Token is valid")
```

### Module: `session_group`

**Path**: `src/mcp/client/session_group.py`

Manages multiple MCP session connections concurrently.

#### Classes

- **`ClientSessionGroup`**: Manages connections to multiple MCP servers.
  - **Methods**:
    - `__init__(self, exit_stack, component_name_hook)`
    - `sessions(self) -> list[mcp.ClientSession]`
    - `prompts(self) -> dict[str, types.Prompt]`
    - `resources(self) -> dict[str, types.Resource]`
    - `tools(self) -> dict[str, types.Tool]`

#### Example

```python
from mcp.client.session_group import ClientSessionGroup

session_group = ClientSessionGroup(exit_stack=None, component_name_hook=None)
sessions = session_group.sessions()
print(sessions)
```

### Module: `streamable_http`

**Path**: `src/mcp/client/streamable_http.py`

Implements the StreamableHTTP transport for MCP clients.

#### Classes

- **`StreamableHTTPError`**: Base exception for StreamableHTTP errors.
- **`ResumptionError`**: Raised for invalid resumption requests.
- **`RequestContext`**: Context for a request operation.

#### Functions

- **`_update_headers_with_session(self, base_headers) -> dict[str, str]`**
- **`_is_initialization_request(self, message) -> bool`**
- **`_is_initialized_notification(self, message) -> bool`**
- **`_maybe_extract_session_id_from_response(self, response) -> None`**

#### Example

```python
from mcp.client.streamable_http import RequestContext

context = RequestContext(url='https://api.example.com', headers={}, timeout=30, sse_read_timeout=60, auth=None)
```

## Error Handling

Common errors include `StreamableHTTPError` and `ResumptionError`. These should be caught and handled appropriately to ensure robust applications.

## Authentication

OAuth2 authentication is handled by the `OAuthClientProvider` class, which supports token storage and automatic refresh.

## Response Formats

Responses are typically JSON objects. Ensure to handle and parse responses correctly to extract necessary information.

This documentation provides a comprehensive overview of the Python SDK, detailing core modules, classes, and functions with practical examples and error handling strategies. For further details, refer to the source code and explore additional modules and utilities.