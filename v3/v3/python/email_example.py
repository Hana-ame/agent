from email.message import EmailMessage
import email.utils

msg = EmailMessage()
msg['From'] = 'sender@example.com'
msg['To'] = 'recipient@example.com'
msg['Subject'] = 'Test Email'
msg['Date'] = email.utils.formatdate()
msg.set_content('This is a test email body.')

# 添加附件
msg.add_attachment(b'Attachment content', maintype='text', subtype='plain', filename='test.txt')

print("Email headers:")
print(f"From: {msg['From']}")
print(f"To: {msg['To']}")
print(f"Subject: {msg['Subject']}")
print(f"Date: {msg['Date']}")
print("Body preview:", msg.get_content())
