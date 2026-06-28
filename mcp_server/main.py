"""MCP Server — stdio transport runner."""

from mcp_server.server import mcp


def main() -> None:
    """Run the MCP server using stdio transport."""
    # stdio is the canonical transport for local co-process MCP servers.
    # LangGraph spawns this server as a subprocess and communicates via pipes.
    # For remote/networked setups, switch to mcp.run(transport="sse").
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
