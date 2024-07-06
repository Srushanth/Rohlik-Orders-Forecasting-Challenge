"""
👨‍💻 Author: Srushanth Baride

📧 Email: Srushanth.Baride@gmail.com

🏢 Organization: 🚀 Rocket ML

📅 Date: 05-July-2024

📚 Description: TODO.
"""

import matplotlib.pyplot as plt

# Sample data
ranks = [168, 168, 168, 168]
submissions = [23, 24, 25, 26]

# Create the bar chart
plt.plot(submissions, ranks, marker="o")
plt.xlabel("Submission")
plt.ylabel("Rank")
plt.title("Improvement in Ranking Over Submissions")
plt.gca().invert_yaxis()  # Invert y-axis to show improvement (lower rank is better)

# Annotate points with ranks
for i, rank in enumerate(ranks):
    plt.text(i, rank, str(rank))

# Save the chart as an image
plt.savefig("../images/rank_submission_chart.png")
