from workflow.qa import RubricsColumn

def format_grade_characteristics(columns: list[str], sources: list[str], table: list[list[str]]):
    characteristics_table = ["Below is the Characteristics table of the above studies.\n"]

    # Table header
    header = "|"
    border = "|"
    for column in columns:
        header += f"{column}|"
        border += f"{'-' * 3}|"
    characteristics_table.append(header)
    characteristics_table.append(border)

    # Table body
    for src, row in zip(sources, table):
        cells = f"|{src}|"
        for cell in row:
            cells += f"{cell}|"
        characteristics_table.append(cells)

    return "\n".join(characteristics_table)

def get_grade_summary(row):
    summary_table = []

    summary_columns = row['summary_title']
    summary_rows = row['summary']

    if len(summary_columns) > 0:
        header = "|"
        border = "|"
        for column in summary_columns:
            header += f"{column}|"
            border += f"{'-' * 3}|"
        summary_table.append(header)
        summary_table.append(border)

        for summary_row in summary_rows:
            cells = "|"
            for cell in summary_row:
                cells += f"{cell}|"
            summary_table.append(cells)

    forest_title = row['forest_title']
    forest = row['forest']

    if len(forest_title) > 0:
        for title, value in zip(forest_title, forest):
            summary_table.append(f"{title}: {value}")

    if len(summary_table) == 0:
        return ""

    return "Below is the summary table of the above studies.\n\n" + "\n".join(summary_table)

def format_quality_table(rubrics: list[RubricsColumn], sources: list[str], scores: list[list[str | int]]):    
    qa_table = ["Below is the quality assessment table of the above studies.\n"]
    
    # Options
    for col in rubrics:
        optionsStr = ", ".join([str(option) for option in col.options])
        qa_table.append(f"{col.title}: {optionsStr}")

    # Table header
    header = "|"
    border = "|"

    for col in rubrics:
        header += f"{col.title}|"
        border += f"{'-' * 3}|"
    qa_table.append(header)
    qa_table.append(border)

    # Table body
    for src, score_row in zip(sources, scores):
        cells = f"|{src}|"
        for score in score_row:
            cells += f"{str(score)}|"
        qa_table.append(cells)

    return "\n".join(qa_table)