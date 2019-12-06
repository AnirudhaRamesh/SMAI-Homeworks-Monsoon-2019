from shiksha import login, shiksha_url
import getpass, lxml.html
username = input('Username: ')
password = getpass.getpass()
session = login(username, password)
response = session.get(shiksha_url+'/asgn/app/course/25/questions/')
table = lxml.html.fromstring(response.text).xpath('//div[@class="button-group small"]')
questions = lxml.html.fromstring(response.text).xpath('//tr')
try:
	questions.pop(0)
except:
	print('wrong username/password')
	exit(0)
tot_marks = 0
no_evaluated = 0
for details in questions:
	name = details[0].text_content();	name = (" ").join(list(filter(None,name.replace('\n','').split(' '))))
	link = details[3][0][2].items()[0];	link = link[1] if link[0]=='href' else None
	my_marks = 0
	if not link:
		my_marks = 'not evaluated'
	else:
		response = session.get(shiksha_url+link)
		marks = lxml.html.fromstring(response.text).xpath('//div[@class="row card-row"]')
		rawmarks = marks[0].text_content().replace(' ','').split('\n')
		rawmarks = list(filter(None, rawmarks))
		if rawmarks[1] == 'Notavailableyet.':
			my_marks = 'not evaluated'
		else:
			no_evaluated += 1
			my_marks = float(rawmarks[1].split(':')[1])
			tot_marks += my_marks
	print("{0:50} : {1}".format(name,my_marks))
print('______________________________________________________________________________')
print("\033[1m{0:50} : {1} from {2} evaluated homeworks".format("Total Marks",tot_marks,no_evaluated))
