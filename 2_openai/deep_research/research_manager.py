from agents import Runner, trace, gen_trace_id
from search_agent import search_agent
from planner_agent import planner_agent, WebSearchItem, WebSearchPlan
from writer_agent import writer_agent, ReportData
from email_agent import email_agent
from clarifying_agent import clarifying_agent
import asyncio

class ResearchManager:

    
    async def run_with_clarified_query(self, clarified_query: dict):
        """ Run the research process with pre-clarified query """
        trace_id = gen_trace_id()
        with trace("Research trace", trace_id=trace_id):
            print(f"View trace: https://platform.openai.com/traces/trace?trace_id={trace_id}")
            yield f"View trace: https://platform.openai.com/traces/trace?trace_id={trace_id}"
            print("Starting research with clarified query...")
            yield "Planning searches with your clarified requirements..."
            search_plan = await self.plan_searches(clarified_query)
            yield "Searches planned, starting to search..."     
            search_results = await self.perform_searches(search_plan)
            yield "Searches complete, writing report..."
            report = await self.write_report(clarified_query['original_query'], search_results)
            yield "Report written, sending email..."
            await self.send_email(report)
            yield "Email sent, research complete"
            yield report.markdown_report
        
    async def clarify_query(self, query: str) -> dict:
        """ Clarify the query """
        print("Clarifying query...")
        result = await Runner.run(
            clarifying_agent,
            query,
        )
        print("Query clarified")
        return {
            "original_query": query,
            "clarifying_questions": result.final_output
        }
    
    async def plan_searches(self, clarified_query: dict) -> WebSearchPlan:
        """ Plan the searches to perform for the query """
        print("Planning searches...")
        input_text = f"Original Query: {clarified_query['original_query']}\n\nClarifying Questions and Analysis:\n{clarified_query['clarifying_questions']}"
        result = await Runner.run(
            planner_agent,
            input_text,
        )
        print(f"Will perform {len(result.final_output.searches)} searches")
        return result.final_output_as(WebSearchPlan)

    async def perform_searches(self, search_plan: WebSearchPlan) -> list[str]:
        """ Perform the searches to perform for the query """
        print("Searching...")
        num_completed = 0
        tasks = [asyncio.create_task(self.search(item)) for item in search_plan.searches]
        results = []
        for task in asyncio.as_completed(tasks):
            result = await task
            if result is not None:
                results.append(result)
            num_completed += 1
            print(f"Searching... {num_completed}/{len(tasks)} completed")
        print("Finished searching")
        return results

    async def search(self, item: WebSearchItem) -> str | None:
        """ Perform a search for the query """
        input = f"Search term: {item.query}\nReason for searching: {item.reason}"
        try:
            result = await Runner.run(
                search_agent,
                input,
            )
            return str(result.final_output)
        except Exception:
            return None

    async def write_report(self, query: str, search_results: list[str]) -> ReportData:
        """ Write the report for the query """
        print("Thinking about report...")
        input = f"Original query: {query}\nSummarized search results: {search_results}"
        result = await Runner.run(
            writer_agent,
            input,
        )

        print("Finished writing report")
        return result.final_output_as(ReportData)
    
    async def send_email(self, report: ReportData) -> None:
        print("Writing email...")
        result = await Runner.run(
            email_agent,
            report.markdown_report,
        )
        print("Email sent")
        return report