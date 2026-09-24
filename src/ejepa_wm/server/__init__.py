"""Standalone MCP servers for the pluggable World Model.

``ewm_predict`` exposes the single-step EWM prediction primitive as an MCP tool so an
agent can call the world model like any other tool (email, calendar). Importing this
package does not pull in ``fastmcp``; import :mod:`ejepa_wm.server.ewm_predict` for that.
"""
