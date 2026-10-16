
from pathlib import Path
import pymupdf

OUTPUT = Path(__file__).parent / "company_policy.pdf"

document = pymupdf.open()

# Page 1
page1 = document.new_page()
page1.insert_text(
    (72, 72),
    "Company Remote Work Policy",
    fontsize=16
)
page1.insert_text(
    (72, 110),
    "Employees can work remotely three days per week."
)

# Page 2
page2 = document.new_page()
page2.insert_text(
    (72, 72),
    "Company Leave Policy",
    fontsize=16
)
page2.insert_text(
    (72, 110),
    "Employees receive twenty days of annual leave."
)

document.save(str(OUTPUT))
document.close()

print(f"Created sample PDF: {OUTPUT}")
