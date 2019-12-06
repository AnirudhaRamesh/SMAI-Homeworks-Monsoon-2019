import requests, lxml.html
shiksha_url = 'http://shiksha.iiit.ac.in'
def login(username, password):
	session = requests.session()
	# visit shiksha portal
	response = session.get(shiksha_url+'/asgn/accounts/login')
	# input credentials and hidden attributes
	login_form = lxml.html.fromstring(response.text)
	hidden_inputs = login_form.xpath('//form//input[@type="hidden"]/@value')
	form = {}
	form['execution'] = hidden_inputs[0]
	form["_eventId"] = hidden_inputs[1]
	form['username'] = username
	form['password'] = password
	# login with credentials
	session.post(response.url, data=form)
	return session