import os
import pandas as pd

class DataLogger:
    def __init__(self):
        pass
    
    # function to log evaluation data to CSV
    def log_evaluation_data(self, eval_type, llm_response, document_names, data_chunks, message, q_id, req_body):
        if eval_type not in ["baseline", "enhanced"]:
            print(f"Invalid evaluation type: {eval_type}. Must be 'baseline' or 'enhanced'.")
            return
        
        try:
            # Use absolute path relative to project root
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            csv_path = os.path.join(project_root, "evaluation", "evaluation_responses.csv")

            # Ensure the evaluation directory exists
            os.makedirs(os.path.dirname(csv_path), exist_ok=True)

            # Check if CSV file exists, if not create it with headers
            if not os.path.exists(csv_path):
                headers = [
                    "Question_ID", "System_Variant", "Change in requst body", "Question_Text", "Reference_Answer",
                    "System_Answer_Text", "Ground_Truth_Doc_Names", "Retrieved_Doc_Names_Ordered",
                    "Retrieved_Chunks_Text", "Human_Relevance_Score_1_5", "Human_Accuracy_Score_1_5",
                    "Human_Completeness_Score_1_5", "Human_Hallucination_Present", "Notes(Not mandatory)"
                ]
                df = pd.DataFrame(columns=headers)
                df.to_csv(csv_path, index=False, encoding='utf-8')

            # Read existing CSV with explicit encoding handling
            try:
                df = pd.read_csv(csv_path, encoding='utf-8')
            except UnicodeDecodeError:
                # Try different encodings if UTF-8 fails
                try:
                    df = pd.read_csv(csv_path, encoding='latin-1')
                except UnicodeDecodeError:
                    # If all else fails, recreate the file
                    print("CSV file appears corrupted, recreating...")
                    headers = [
                        "Question_ID", "System_Variant","Change in requst body", "Question_Text", "Reference_Answer",
                        "System_Answer_Text", "Ground_Truth_Doc_Names", "Retrieved_Doc_Names_Ordered",
                        "Retrieved_Chunks_Text", "Human_Relevance_Score_1_5", "Human_Accuracy_Score_1_5",
                        "Human_Completeness_Score_1_5", "Human_Hallucination_Present", "Notes(Not mandatory)"
                    ]
                    df = pd.DataFrame(columns=headers)

            # Clean the text data to ensure it's properly encoded
            def clean_text(text):
                if isinstance(text, str):
                    # Remove or replace problematic characters
                    return text.encode('utf-8', errors='ignore').decode('utf-8')
                elif text is None:
                    return ""
                return str(text)

            # Handle llm_response - support both dict and string
            if isinstance(llm_response, dict):
                answer_text = llm_response.get("answer", "No answer generated.")
            else:
                answer_text = str(llm_response)

            # Find the row where Question_Text = message & System_Variant = eval_type
            mask = (df['Question_ID'] == q_id) & (df['System_Variant'] == eval_type)
            matching_rows = df[mask]

            if len(matching_rows) > 0:
                # Update existing row
                row_index = matching_rows.index[0]
                
                df.at[row_index, 'Question_Text'] = clean_text(message)
                df.at[row_index, 'Change in request body'] = clean_text(str(req_body))
                df.at[row_index, 'System_Answer_Text'] = clean_text(answer_text)
                df.at[row_index, 'Retrieved_Doc_Names_Ordered'] = clean_text(" | ".join(document_names) if document_names else "")
                df.at[row_index, 'Retrieved_Chunks_Text'] = clean_text(" | ".join(data_chunks) if data_chunks else "")
                print(f"Updated existing row {row_index} for {eval_type} evaluation")
            else:
                # Create new row
                new_row = {
                    'Question_ID': q_id,
                    'System_Variant': eval_type,
                    'Question_Text': clean_text(message),
                    'Reference_Answer': '',  # Will be filled manually
                    'System_Answer_Text': clean_text(answer_text),
                    'Ground_Truth_Doc_Names': '',  # Will be filled manually
                    'Retrieved_Doc_Names_Ordered': clean_text(" | ".join(document_names) if document_names else ""),
                    'Retrieved_Chunks_Text': clean_text(" | ".join(data_chunks) if data_chunks else ""),
                    'Human_Relevance_Score_1_5': '',  # Will be filled manually
                    'Human_Accuracy_Score_1_5': '',  # Will be filled manually
                    'Human_Completeness_Score_1_5': '',  # Will be filled manually
                    'Human_Hallucination_Present': '',  # Will be filled manually
                    'Notes(Not mandatory)': ''  # Will be filled manually
                }

                # Add new row to dataframe
                df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
                print(f"Created new row for {eval_type} evaluation")

            # Save back to CSV with explicit UTF-8 encoding
            df.to_csv(csv_path, index=False, encoding='utf-8')
            print(f"Successfully logged evaluation data to {csv_path}")

        except Exception as eval_error:
            print(f"Evaluation Logging Error: {eval_error}")
            import traceback
            traceback.print_exc()
