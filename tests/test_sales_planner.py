import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from app.agents.sales import mcp_agent


class SalesPlannerTests(unittest.IsolatedAsyncioTestCase):
    def test_bare_sales_order_request_only_collects_details(self):
        result = mcp_agent._initial_document_details_request("create a new sales order")
        self.assertIsNotNone(result)
        answer, presentation = result
        self.assertIn("preview", answer)
        self.assertEqual(presentation["title"], "Sales Order details needed")
        self.assertEqual(presentation["fields"][0]["label"], "Customer")
        self.assertEqual(presentation["fields"][1]["label"], "Items")

    def test_bare_quotation_request_only_collects_details(self):
        result = mcp_agent._initial_document_details_request("create a new sales quotation")
        self.assertIsNotNone(result)
        _, presentation = result
        self.assertEqual(presentation["title"], "Sales Quotation details needed")
        self.assertEqual(presentation["fields"][2]["label"], "Valid till date (optional)")

    def test_quotation_details_keep_document_tools(self):
        tools = [SimpleNamespace(name=name) for name in (
            "prepare_quotation", "prepare_sales_order", "search_customers", "prepare_customer", "search_items"
        )]
        state = {
            "user_message": "customer name - Umesh parekh\nitem - Dell Mouce\nquantity - 1500 nos\nrate - 151\nvalid till - 31 dec 2026",
            "conversation_history": [
                {"role": "user", "content": "please create a sales quotation"},
                {"role": "assistant", "content": "Please provide customer and items."},
            ],
        }
        self.assertEqual([t.name for t in mcp_agent._request_entity_tools(state, tools)], ["prepare_quotation"])
        state["user_message"] = "please create a sales order"
        self.assertEqual([t.name for t in mcp_agent._request_entity_tools(state, tools)], ["prepare_sales_order"])

    async def test_prompt_only_names_available_metadata(self):
        llm = SimpleNamespace(bind_tools=unittest.mock.Mock())
        invocation = AsyncMock(return_value=SimpleNamespace(content="Which customer?", tool_calls=[]))
        llm.bind_tools.return_value.ainvoke = invocation
        with patch.object(mcp_agent, "get_llm", return_value=llm):
            await mcp_agent._plan_tool_call(
                "create a sales quotation", [SimpleNamespace(name="prepare_quotation")]
            )
        prompt = invocation.call_args.args[0][0]["content"]
        self.assertNotIn("get_required_fields", prompt)
        self.assertNotIn("get_doctype_schema", prompt)

    async def test_entity_filter_uses_only_the_requested_document_tool(self):
        tools = [SimpleNamespace(name=name) for name in (
            "prepare_quotation", "prepare_sales_order", "get_required_fields"
        )]
        for entity, expected in (("quotation", "prepare_quotation"), ("order", "prepare_sales_order")):
            with self.subTest(entity=entity), patch.dict("os.environ", {
                "SALES_MCP_ENABLED": "true", "ERP_MCP_URL": "http://test/mcp",
                "ERP_MCP_SHARED_SECRET": "test",
            }), patch("langchain_mcp_adapters.client.MultiServerMCPClient") as client, patch.object(
                mcp_agent, "_plan_tool_call", new_callable=AsyncMock,
                return_value=(None, {}, "Which customer?"),
            ) as planner:
                client.return_value.get_tools = AsyncMock(return_value=tools)
                await mcp_agent.sales_mcp_node({"user_message": f"please create a sales {entity}"})
                self.assertEqual({t.name for t in planner.call_args.args[1]}, {expected})

    async def test_detail_follow_up_stays_with_the_requested_sales_order(self):
        tools = [SimpleNamespace(name=name) for name in (
            "prepare_sales_order", "prepare_quotation", "prepare_customer", "prepare_item"
        )]
        with patch.dict("os.environ", {
            "SALES_MCP_ENABLED": "true", "ERP_MCP_URL": "http://test/mcp",
            "ERP_MCP_SHARED_SECRET": "test",
        }), patch("langchain_mcp_adapters.client.MultiServerMCPClient") as client, patch.object(
            mcp_agent, "_plan_tool_call", new_callable=AsyncMock,
            return_value=(None, {}, "Which quantity?"),
        ) as planner:
            client.return_value.get_tools = AsyncMock(return_value=tools)
            await mcp_agent.sales_mcp_node({
                "user_message": "Customer - Umesh Parekh\nItems - ggg\nDelivery date - 31 Dec 2027",
                "conversation_history": [
                    {"role": "user", "content": "create a sales order"},
                    {"role": "assistant", "content": "Please provide the order details."},
                ],
            })
        self.assertEqual([tool.name for tool in planner.call_args.args[1]], ["prepare_sales_order"])

    async def test_provider_failure_is_not_reported_as_connectivity(self):
        with patch.dict("os.environ", {
            "SALES_MCP_ENABLED": "true", "ERP_MCP_URL": "http://test/mcp",
            "ERP_MCP_SHARED_SECRET": "test",
        }), patch("langchain_mcp_adapters.client.MultiServerMCPClient") as client, patch.object(
            mcp_agent, "_plan_tool_call", new_callable=AsyncMock,
            side_effect=ValueError("tool_use_failed"),
        ), self.assertLogs(mcp_agent.logger, level="ERROR"):
            client.return_value.get_tools = AsyncMock(return_value=[SimpleNamespace(name="prepare_quotation")])
            result = await mcp_agent.sales_mcp_node({"user_message": "create a quotation"})
        self.assertIn("AI could not process", result["answer"])
        self.assertNotIn("could not be reached", result["answer"])
        self.assertFalse(result["tool_executed"])


if __name__ == "__main__":
    unittest.main()
