import logging
import os # Good practice to import os if dealing with file paths

def save_output(filename: str, output_lines: list[str]):
    """Saves the output lines to a file.

    Args:
        filename (str): The name of the file to save the output to.
        output_lines (list): A list of strings, where each string is a line to write to the file.
    """
    if not output_lines:
        logging.warning(f"No output lines to save to '{filename}'. Skipping file creation.")
        return

    try:
        # Ensure the directory exists if filename includes a path
        output_dir = os.path.dirname(filename)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            logging.info(f"Created output directory: {output_dir}")

        with open(filename, "w", encoding="utf-8") as f:
            f.write("\n".join(output_lines))
        logging.info(f"✅ Meta Minds output successfully saved to '{filename}'")
    except IOError as e:
        logging.error(f"IOError occurred while saving output to '{filename}': {e}")
        # Depending on severity, you might re-raise or handle differently
    except Exception as e:
        logging.error(f"An unexpected error occurred while saving output to '{filename}': {e}")
        # Depending on severity, you might re-raise or handle differently