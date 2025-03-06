import gradio as gr
from utils import (
    process_single_text_with_products,
    process_csv_file_with_products,
    display_product_info
)

with gr.Blocks() as demo:
    gr.Markdown("# CHIPP-AI")

    with gr.Tabs():
        with gr.TabItem("Single Text Analysis"):
            with gr.Row():
                input_text = gr.Textbox(
                    label="Enter Text Reviews",
                    placeholder="Type your text reviews here...",
                    lines=5
                )
            with gr.Row():
                btn_text = gr.Button("Review Review!")
            with gr.Row():
                out_sentiment = gr.Textbox(label="Predicted Sentiment")
            with gr.Row():
                out_conf_plot = gr.Plot(label="Confidence Pie Chart")
            with gr.Row():
                product_dropdown_text = gr.Dropdown(label="Select Product", choices=[], interactive=True)
            product_wc_state_text = gr.State()
            with gr.Row():
                out_topics_text = gr.Textbox(label="Product Topics", lines=10)

            btn_text.click(
                process_single_text_with_products,
                inputs=input_text,
                outputs=[out_sentiment, out_conf_plot, product_dropdown_text, product_wc_state_text]
            )
            # For single text analysis, only display topics text
            product_dropdown_text.change(
                fn=lambda x, y: display_product_info(x, y)[0],
                inputs=[product_dropdown_text, product_wc_state_text],
                outputs=out_topics_text
            )

        with gr.TabItem("CSV Analysis"):
            with gr.Row():
                csv_file = gr.File(label="Upload CSV File", file_types=['.csv'], type="filepath")
                col_name = gr.Textbox(
                    label="Column Name",
                    placeholder="Enter column name containing text (or leave blank for default)"
                )
            with gr.Row():
                btn_csv = gr.Button("Review Reviews!")
            with gr.Row():
                out_sentiment_csv = gr.Plot(label="Sentiment Distribution Pie Chart")
            with gr.Row():
                product_dropdown_csv = gr.Dropdown(label="Select Product", choices=[], interactive=True)
            product_wc_state_csv = gr.State()
            with gr.Row():
                out_topics_csv = gr.Textbox(label="Product Topics", lines=10)
            with gr.Row():
                out_pos_wc_csv = gr.Plot(label="Positive Word Cloud")
                out_neg_wc_csv = gr.Plot(label="Negative Word Cloud")

            btn_csv.click(
                process_csv_file_with_products,
                inputs=[csv_file, col_name],
                outputs=[out_sentiment_csv, product_dropdown_csv, product_wc_state_csv]
            )
            product_dropdown_csv.change(
                display_product_info,
                inputs=[product_dropdown_csv, product_wc_state_csv],
                outputs=[out_topics_csv, out_pos_wc_csv, out_neg_wc_csv]
            )

demo.launch(debug=True)
