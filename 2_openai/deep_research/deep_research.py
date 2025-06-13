import gradio as gr
from dotenv import load_dotenv
from research_manager import ResearchManager

load_dotenv(override=True)

research_manager = ResearchManager()

async def get_clarifying_questions(query: str):
    """Get clarifying questions for the user's query"""
    if not query.strip():
        return (
            gr.update(value="", visible=False),  # clarifying_questions_display
            gr.update(visible=False),  # clarifying_section
            gr.update(visible=False),  # answer1
            gr.update(visible=False),  # answer2
            gr.update(visible=False),  # answer3
            gr.update(visible=False)   # start_research_button
        )
    
    clarified_result = await research_manager.clarify_query(query)
    clarifying_questions = clarified_result['clarifying_questions']
    
    return (
        gr.update(value=clarifying_questions, visible=True),  # clarifying_questions_display
        gr.update(visible=True),  # clarifying_section
        gr.update(visible=True),  # answer1
        gr.update(visible=True),  # answer2
        gr.update(visible=True),  # answer3
        gr.update(visible=True)   # start_research_button
    )

async def run_research_with_answers(query: str, answer1: str, answer2: str, answer3: str):
    """Run research with the user's answers to clarifying questions"""
    # Combine the original query with user answers
    enhanced_query = f"""Original Query: {query}

User's Clarifying Responses:
1. {answer1}
2. {answer2}
3. {answer3}"""
    
    # Create a mock clarified query structure to maintain compatibility
    clarified_query = {
        "original_query": query,
        "clarifying_questions": enhanced_query
    }
    
    # Run the research process starting from planning searches
    async for chunk in research_manager.run_with_clarified_query(clarified_query):
        yield chunk

with gr.Blocks(theme=gr.themes.Default(primary_hue="sky")) as ui:
    gr.Markdown("# Deep Research")
    
    with gr.Column():
        query_textbox = gr.Textbox(
            label="What topic would you like to research?",
            placeholder="Enter your research question here..."
        )
        get_questions_button = gr.Button("Get Clarifying Questions", variant="secondary")
        
        # Clarifying questions section (initially hidden)
        clarifying_questions_display = gr.Markdown(
            label="Clarifying Questions",
            visible=False
        )
        
        with gr.Column(visible=False) as clarifying_section:
            gr.Markdown("### Please answer these clarifying questions to improve your research:")
            answer1 = gr.Textbox(label="Answer to Question 1", lines=2)
            answer2 = gr.Textbox(label="Answer to Question 2", lines=2)
            answer3 = gr.Textbox(label="Answer to Question 3", lines=2)
            start_research_button = gr.Button("Start Research", variant="primary")
        
        # Results section
        report = gr.Markdown(label="Research Report")
    
    # Event handlers
    get_questions_button.click(
        fn=get_clarifying_questions,
        inputs=[query_textbox],
        outputs=[clarifying_questions_display, clarifying_section, answer1, answer2, answer3, start_research_button]
    )
    
    start_research_button.click(
        fn=run_research_with_answers,
        inputs=[query_textbox, answer1, answer2, answer3],
        outputs=[report]
    )

ui.launch(inbrowser=True)

"""
1. Update research assistant to ask clarifying questions before starting the search
2. Make the search tool agentic
3. Add handoff to writer agent
4. Add handoff to email agent
5. Add guardrails to prevent the agent from hallucinating
"""