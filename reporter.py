from fpdf import FPDF
import base64

class PDF(FPDF):
    def header(self):
        self.set_font('Arial','B',12)
        self.cell(0,10,'Deep Research Agent Report',0,1,'C')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial','I',8)
        self.cell(0,10,f'Page {self.page_no()}',0,0,'C')
    
def create_pdf(topic,trends,gaps,roadmap,summary):
    pdf = PDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True,margin=15)
    pdf.set_font("Arial",size=12)

    #heper fn to add section
    def add_section(title,content):
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, title, 0, 1, 'L')
        pdf.set_font("Arial", size=11)
        # Handle encoding for special characters
        safe_content = content.encode('latin-1', 'replace').decode('latin-1')
        pdf.multi_cell(0, 7, safe_content)
        pdf.ln(5)
        #content
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(0, 10, f"Research Topic: {topic}", 0, 1, 'L')
        pdf.ln(5)

        if summary: add_section("Executive Summary", summary)
        if trends: add_section("Key Research Trends", trends)

        if gaps:
            pdf.set_font("Arial",'B',14)
            pdf.cell(0,10,"Identified Research Gaps",0,1,'L')
            pdf.set_font("Arial",size=11)
            for g in gaps:
                txt = f"- {g['source']}:{g['gaps']}"
                pdf.multi_cell(0,7,txt.encode('lattin-1','replace').decode('latin-1'))


            pdf.ln(5)

        if roadmap: add_section("Proposed Roadmap",roadmap)

def get_download_link(pdf_data,filename="Research_Report.pdf"):
    b64 = base64.b64encode(pdf_data).decode()
    return f'<a href="data:application/octet-stream;base64,{b64}" download="{filename}" style="background-color:#ef4444; color:white; padding:10px 20px; text-decoration:none; border-radius:5px; font-weight:bold;">📥 Download Full Report (PDF)</a>'