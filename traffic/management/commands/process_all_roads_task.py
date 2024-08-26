from django.core.management.base import BaseCommand
import subprocess
import os

class Command(BaseCommand):
    help = 'Run the Django background task worker'

    def handle(self, *args, **kwargs):
        self.stdout.write("Starting the task worker...")

        # Define the path to your Python executable within the virtual environment

        # Run the background task worker using the virtual environment's Python
        subprocess.call(['python', 'manage.py', 'process_tasks'])
