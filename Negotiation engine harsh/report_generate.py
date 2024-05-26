from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, HRFlowable
from reportlab.lib import colors
from reportlab.lib.units import inch

def generate_pdf_report(data, filename):
    doc = SimpleDocTemplate(filename, pagesize=A4)
    styles = getSampleStyleSheet()

    logo = "Negotiation engine harsh/static/Screenshot 2024-05-26 at 05.12.32.png"
    logo_image = Image(logo, width=2*inch, height=2*inch)
    logo_image.hAlign = 'CENTER'

    title_style = ParagraphStyle(
        name='Title',
        fontSize=24,
        leading=30,
        alignment=1,  
        spaceAfter=20,
        fontName='Helvetica-Bold',
        textColor=colors.HexColor('#000080')
    )
    title = Paragraph("Negotiation Summary Report", title_style)

    subtitle_style = ParagraphStyle(
        name='Subtitle',
        fontSize=14,
        leading=18,
        alignment=1,
        spaceAfter=20,
        fontName='Helvetica',
        textColor=colors.HexColor('#666666')
    )
    subtitle = Paragraph("Generated by Negotiation AI Assistant", subtitle_style)

    line = HRFlowable(width="100%", thickness=1, color=colors.HexColor('#000080'), spaceBefore=10, spaceAfter=20)


    summary_title_style = ParagraphStyle(
        name='SummaryTitle',
        fontSize=18,
        leading=22,
        spaceAfter=10,
        fontName='Helvetica-Bold',
        textColor=colors.HexColor('#000080')
    )
    summary_title = Paragraph("Summary", summary_title_style)

    summary_style = ParagraphStyle(
        name='Summary',
        fontSize=12,
        leading=18,
        spaceAfter=20,
        fontName='Helvetica'
    )
    summary = Paragraph(data['summary'], summary_style)

    status_title = Paragraph("Status", summary_title_style)
    status = Paragraph(data['status'], summary_style)
    
    compromise_title = Paragraph("Last Proposed Compromise", summary_title_style)
    compromise = Paragraph(data['last_proposed_compromise'], summary_style)

    footer_style = ParagraphStyle(
        name='Footer',
        fontSize=10,
        leading=12,
        alignment=1,
        spaceBefore=20,
        fontName='Helvetica-Oblique',
        textColor=colors.HexColor('#666666')
    )
    footer = Paragraph("Confidential - For internal use only", footer_style)

    elements = [
        logo_image,
        Spacer(1, 20),
        title, 
        subtitle, 
        line,
        summary_title, 
        summary, 
        status_title, 
        status, 
        compromise_title, 
        compromise,
        Spacer(1, 24),
        line,
        footer
    ]
    doc.build(elements)


if __name__ == "__main__":
    data = {
        "summary": "The negotiation focused on price and delivery time, with both parties seeking a fair agreement.",
        "status": "In Progress",
        "last_proposed_compromise": "Both parties agreed to split the delivery cost and reduce the price by 5%."
    }

    # Generate the PDF report
    generate_pdf_report(data, "Negotiation_Summary_Report.pdf")