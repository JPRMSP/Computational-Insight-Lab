import streamlit as st
import numpy as np
import sympy as sp
import random
import matplotlib.pyplot as plt
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter
from io import BytesIO
import base64
from matplotlib import animation

st.set_page_config(layout="wide")

# -------------------------------------------------------------------
# Utility Functions
# -------------------------------------------------------------------

def is_stochastic_matrix(P):
    return np.allclose(P.sum(axis=1), 1) and np.all(P >= 0)

def stationary_distribution(P):
    eigvals, eigvecs = np.linalg.eig(P.T)
    idx = np.argmin(np.abs(eigvals - 1))
    v = np.real(eigvecs[:, idx])
    return v / np.sum(v)

def generate_latex_steps_stationary(P):
    n = len(P)
    s = "\\textbf{Stationary Distribution Derivation:}\n\nSolve: $\\pi P = \\pi$ and $\\sum_i \\pi_i = 1$.\n\n"
    for i in range(n):
        eq = " + ".join([f"\\pi_{j} P_{{{j}{i}}}" for j in range(n)])
        s += f"$\\pi_{i} = {eq}$\n\n"
    s += "$\\sum_i \\pi_i = 1$"
    return s

def generate_ieee_paper(P, sd, queue_summary, rw_summary, jordan_summary):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("<b>Research Paper: Stochastic Processes Simulation Platform</b>", styles['Title']))
    story.append(Spacer(1, 12))

    story.append(Paragraph("<b>Abstract</b>", styles['Heading2']))
    story.append(Paragraph(
        "This paper presents a unified research platform implementing Markov Chains, Random Walks, "
        "Queueing Theory, and Linear Algebra transformations using purely mathematical formulations.",
        styles['Normal']
    ))
    story.append(Spacer(1, 12))

    story.append(Paragraph("<b>1. Introduction</b>", styles['Heading2']))
    story.append(Paragraph(
        "This work demonstrates a dataset-free computational research suite integrating stochastic processes "
        "and algebraic methods for teaching and research in computing mathematics.",
        styles['Normal']
    ))

    story.append(Paragraph("<b>2. Markov Chain Analysis</b>", styles['Heading2']))
    story.append(Paragraph(f"Transition Matrix: {P}", styles['Normal']))
    story.append(Paragraph(f"Stationary Distribution: {sd}", styles['Normal']))

    story.append(Paragraph("<b>3. Queueing Theory Analysis</b>", styles['Heading2']))
    story.append(Paragraph(queue_summary, styles['Normal']))

    story.append(Paragraph("<b>4. Random Walk Analysis</b>", styles['Heading2']))
    story.append(Paragraph(rw_summary, styles['Normal']))

    story.append(Paragraph("<b>5. Jordan Form Analysis</b>", styles['Heading2']))
    story.append(Paragraph(jordan_summary, styles['Normal']))

    story.append(Paragraph("<b>Conclusion</b>", styles['Heading2']))
    story.append(Paragraph("This paper establishes the feasibility of simulation-driven mathematical research "
                          "without external data.", styles['Normal']))

    doc.build(story)
    pdf = buffer.getvalue()
    buffer.close()
    return pdf

def random_walk_3d_gif(steps=500):
    xs = [0]; ys = [0]; zs=[0]
    x=y=z=0

    for _ in range(steps):
        dx, dy, dz = random.choice([(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)])
        x+=dx; y+=dy; z+=dz
        xs.append(x); ys.append(y); zs.append(z)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    line, = ax.plot([],[],[], lw=2)

    def update(i):
        line.set_data(xs[:i], ys[:i])
        line.set_3d_properties(zs[:i])
        return line,

    ani = animation.FuncAnimation(fig, update, frames=len(xs), interval=20)
    buf = BytesIO()
    ani.save(buf, writer="pillow", format="gif")
    return buf.getvalue()

def optimal_servers(lmbda, mu, cost_cust, cost_server, max_servers=10):
    costs = []
    for c in range(1, max_servers+1):
        rho = lmbda / (c * mu)
        if rho >= 1:
            costs.append(float('inf'))
        else:
            Lq = (rho**(c+1)) / (c * (1-rho)**2)
            cost = cost_cust * Lq + cost_server * c
            costs.append(cost)
    return costs

# -------------------------------------------------------------------
# Streamlit Research Tabs
# -------------------------------------------------------------------

tabs = st.tabs([
    "IEEE Paper Generator",
    "LaTeX Proof Engine",
    "3D Random Walk GIF",
    "Queue Optimization",
    "Full Research Report"
])

# ---------------------------------------------------------------
# 1. IEEE PAPER GENERATOR
# ---------------------------------------------------------------
with tabs[0]:
    st.header("📄 Auto-Generate IEEE Research Paper")

    P_text = st.text_area("Transition Matrix P:")
    queue_sum = st.text_input("Queue Analysis Summary:")
    rw_sum = st.text_input("Random Walk Summary:")
    jordan_sum = st.text_input("Jordan Form Summary:")

    if P_text:
        P = np.array([list(map(float, r.split())) for r in P_text.split(";")])
        sd = stationary_distribution(P)

        if st.button("Generate IEEE PDF"):
            pdf = generate_ieee_paper(P, sd, queue_sum, rw_sum, jordan_sum)
            b64 = base64.b64encode(pdf).decode()
            st.download_button("📥 Download Research Paper", data=pdf, file_name="research_paper.pdf")

# ---------------------------------------------------------------
# 2. LATEX PROOF ENGINE
# ---------------------------------------------------------------
with tabs[1]:
    st.header("🧠 LaTeX Mathematical Derivation Engine")

    P_text = st.text_area("Enter Markov transition matrix for derivation:")

    if P_text:
        P = np.array([list(map(float, r.split())) for r in P_text.split(";")])
        latex = generate_latex_steps_stationary(P)
        st.code(latex, language="latex")

# ---------------------------------------------------------------
# 3. 3D RANDOM WALK GIF EXPORT
# ---------------------------------------------------------------
with tabs[2]:
    st.header("🌌 3D Random Walk Animation (GIF Export)")

    steps = st.slider("Steps", 100, 3000, 500)

    if st.button("Generate 3D Walk GIF"):
        gif = random_walk_3d_gif(steps)
        st.image(gif)
        st.download_button("📥 Download GIF", data=gif, file_name="3d_random_walk.gif")

# ---------------------------------------------------------------
# 4. QUEUE OPTIMIZATION ENGINE
# ---------------------------------------------------------------
with tabs[3]:
    st.header("⚙️ Optimal Number of Servers in M/M/c Queue")

    lam = st.number_input("Arrival rate λ", 0.1, 10.0, 2.0)
    mu = st.number_input("Service rate μ", 0.1, 10.0, 3.0)
    cost_c = st.number_input("Cost per waiting customer", 1.0, 1000.0, 10.0)
    cost_s = st.number_input("Cost per server", 1.0, 1000.0, 5.0)

    if st.button("Compute Optimal Servers"):
        costs = optimal_servers(lam, mu, cost_c, cost_s)
        fig, ax = plt.subplots()
        ax.plot(range(1, len(costs)+1), costs)
        ax.set_xlabel("Servers (c)")
        ax.set_ylabel("Total Cost")
        st.pyplot(fig)

# ---------------------------------------------------------------
# 5. Full Multi-Section Research Report
# ---------------------------------------------------------------
with tabs[4]:
    st.header("📘 Full Research Report Generator")

    if st.button("Generate Report PDF"):
        pdf = generate_ieee_paper([[0.5,0.5],[0.2,0.8]], [0.285,0.714],
                                  "Sample Queue Summary", "Sample RW Summary",
                                  "Sample Jordan Summary")
        st.download_button("📥 Download Full Research Report", data=pdf, file_name="research_report.pdf")
