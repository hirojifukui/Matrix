from flask_wtf import FlaskForm
from flask_login import current_user
from flask_wtf.file import FileField, FileAllowed, FileRequired
from wtforms import StringField, PasswordField, SubmitField, BooleanField, DateField, MultipleFileField, SelectField, validators
from wtforms.fields.html5 import EmailField
from wtforms.validators import DataRequired, Length, Email, EqualTo, ValidationError
from app.models import User
from datetime import datetime

class RegistrationForm(FlaskForm):
	username = StringField('Username', validators = [DataRequired(), Length(min=5, max=18)])
	email = StringField('Email', validators=[DataRequired(), Email()])
	password = PasswordField('Password', validators=[DataRequired(), Length(min=5, max=18)])
	confirm_password = PasswordField('Confirm Password', validators=[DataRequired(), EqualTo('password')])
	submit = SubmitField('Sign Up')

	def validate_username(self, username):
		user = User.query.filter_by(username=username.data).first()
		if user:
			raise ValidationError('That username is taken. Please choose a different one.')

	def validate_email(self, email):
		user = User.query.filter_by(email=email.data).first()
		if user:
			raise ValidationError('That email is taken. Please choose a different one.')

class ChildForm(FlaskForm):
	GENDER_CHOICES = [('male','Male'), ('female','Female')]
	GRADE_CHOICES = [('pk','Pre-K'), ('k','Kindergarten'), ('1','Grade-1'), ('2','Grade-2'), ('3','Grade-3'), ('4','Grade-4'), ('5','Grade-5'), ('6','Grade-6')]
	cfirst = StringField('First name', validators = [DataRequired(), Length(min=1, max=18)])
	clast = StringField('Lastt name', validators = [DataRequired(), Length(min=1, max=18)])
	cgender = SelectField('Gender', choices = GENDER_CHOICES, validators = [DataRequired()])
	cgrade = SelectField('Grade', choices = GRADE_CHOICES, validators = [DataRequired()])
#	email = EmailField('Email', validators=[validators.Email()])
	submit = SubmitField('Add Child')


	def validate_cbirthday(self, cbirthday):
		try:
			datetime.strptime(str(cbirthday), '%Y-%m-%d')
		except ValueError:
			raise ValueError("Incorrect data format, should be YYYY-MM-DD")
		bdate = str(cbirthday).split("-")
		year = datetime.date.today().year
		if int(bdate[0]) > year - 2 or int(bdate[0]) < year - 20:
			raise ValidationError('Birth year is out of our service range.')

class ProgressForm(FlaskForm):
	child = SelectField('Child', coerce = int, validators=[DataRequired()])
	worksheet = SelectField('Worksheet', coerce=int, validators=[DataRequired()])
	submit = SubmitField('Reload')

class LoginForm(FlaskForm):
	email = StringField('Email', validators=[DataRequired(), Email()])
	password = PasswordField('Password', validators=[DataRequired(), Length(min=5, max=18)])
	remember = BooleanField('Remember Me')
	submit = SubmitField('Login')

class UpdateAccountForm(FlaskForm):
	username = StringField('Username', validators = [DataRequired(), Length(min=5, max=18)])
	email = StringField('Email', validators=[DataRequired(), Email()])
	picture = FileField ('Update Profile Picture', validators=[FileAllowed(['jpg', 'png'])])
	zipcode = StringField('Zip Code', validators = [Length(min=5, max=18)])
	submit = SubmitField('Update')

	def validate_username(self, username):
		if username.data != current_user.username:
			user = User.query.filter_by(username=username.data).first()
			if user:
				raise ValidationError('That username is taken. Please choose a different one.')

	def validate_email(self, email):
		if email.data != current_user.email:
			user = User.query.filter_by(email=email.data).first()
			if user:
				raise ValidationError('That email is taken. Please choose a different one.')

class RequestResetForm(FlaskForm):
    email = StringField('Email',
                        validators=[DataRequired(), Email()])
    submit = SubmitField('Request Password Reset')

    def validate_email(self, email):
        user = User.query.filter_by(email=email.data).first()
        if user is None:
            raise ValidationError('There is no account with that email. You must register first.')


class ResetPasswordForm(FlaskForm):
    password = PasswordField('Password', validators=[DataRequired()])
    confirm_password = PasswordField('Confirm Password',
                                     validators=[DataRequired(), EqualTo('password')])
    submit = SubmitField('Reset Password')
	
class TestDateForm(FlaskForm):
	#child = SelectField('Child', coerce = int, validators=[DataRequired()])
	testdate = DateField('Test Date:', format='%m/%d/%Y', validators=[DataRequired()])
	submit = SubmitField('Set Test Date')
	
class UploadForm(FlaskForm):
    #tdate = DateField('Test Date:', validators=[DataRequired()])
    file = MultipleFileField('Upload Files:', validators=[DataRequired()])
    submit = SubmitField('Upload')
    submit_f = SubmitField('Finish')
	#file = MultipleFileField('Upload Files:', validators =[FileRequired(), FileAllowed(['jpg','jpeg','png','gif'], 'Images only!')])



