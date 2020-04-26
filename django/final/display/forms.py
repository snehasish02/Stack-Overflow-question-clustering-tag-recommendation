from django import forms


class q_form(forms.Form):
    question = forms.CharField(label='question')
