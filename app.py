from flask import Flask, render_template
import flask
import pickle
import pandas as pd

with open(f'model/prediction_18032021.pkl', 'rb') as f:
    model = pickle.load(f)


app = Flask(__name__)

Clg_List = {1: ("Massachusetts Institute of Technology",
                "Stanford University",
                "University of California--Berkeley",
                "California Institute of Technology",
                "Carnegie Mellon University",
                "Purdue University--West Lafayette",
                "University of Michigan--Ann Arbor",
                "Georgia Institute of Technology",
                "University of California--San Diego (Jacobs)",
                "University of Illinois--Urbana-Champaign"
                ),
            2: ("Texas A&M University--College Station",
                "Cornell University",
                "University of Southern California (Viterbi)",
                "University of Texas--Austin (Cockrell)",
                "Columbia University (Fu Foundation)",
                "University of California--Los Angeles (Samueli)",
                "Johns Hopkins University (Whiting)",
                "University of Pennsylvania",
                "Northwestern University (McCormick)",
                "University of Maryland--College Park (Clark)",
                "Harvard University",
                "Princeton University",
                "University of Washington",
                "Duke University (Pratt)",
                "North Carolina State University",
                "University of California--Santa Barbara",
                "University of Colorado--Boulder",
                "University of Wisconsin--Madison",
                "Rice University (Brown)",
                "Ohio State University"
                ),
            3: ("Northeastern University",
                "Virginia Tech",
                "Pennsylvania State University--University Park",
                "University of California--Davis",
                "University of Minnesota--Twin Cities",
                "Boston University",
                "New York University (Tandon)",
                "University of California--Irvine (Samueli)",
                "University of Virginia",
                "Yale University",
                "Arizona State University (Fulton)",
                "Vanderbilt University",
                "Rensselaer Polytechnic Institute",
                "University of Rochester (Hajim)",
                "University of Dayton",
                "University of Florida",
                "Iowa State University",
                "University of Delaware",
                "University of Notre Dame",
                "University of Pittsburgh (Swanson)",
                "Washington University in St. Louis",
                "Case Western Reserve University",
                "Dartmouth College (Thayer)",
                "Brown University",
                "Colorado School of Mines",
                "Rutgers University--New Brunswick",
                "University of Massachusetts--Amherst",
                "University of Utah",
                "Michigan State University",
                "University at Buffalo--SUNY"
                ),
            4: ("University of Tennessee--Knoxville (Tickle)",
                "Stony Brook University--SUNY",
                "University of Arizona",
                "University of Texas--Dallas (Jonsson)",
                "Tufts University",
                "Auburn University (Ginn)",
                "Colorado State University (Scott)",
                "Lehigh University (Rossin)",
                "Rochester Institute of Technology (Gleason)",
                "University of Houston (Cullen)",
                "Clemson University",
                "Oregon State University",
                "University of Central Florida",
                "University of Connecticut",
                "George Washington University",
                "University of North Carolina--Chapel Hill",
                "Washington State University",
                "Wichita State University",
                "Drexel University",
                "Stevens Institute of Technology (Schaefer)",
                "Syracuse University",
                "University of California--Riverside (Bourns)",
                "University of Illinois--Chicago",
                "Missouri University of Science & Technology--Rolla",
                "University of Iowa",
                "University of New Mexico",
                "University of Texas--Arlington",
                "Illinois Institute of Technology (Armour)",
                "Michigan Technological University",
                "New Jersey Institute of Technology",
                "University of California--Santa Cruz (Baskin)",
                "University of Cincinnati",
                "George Mason University (Volgenau)",
                "Mississippi State University (Bagley)",
                "Naval Postgraduate School",
                "University of Nebraska--Lincoln",
                "Worcester Polytechnic Institute",
                "Florida A&M University - Florida State University",
                "Texas Tech University (Whitacre)",
                "University of South Florida"
                ),
            5: ("Brigham Young University",
                "Embry-Riddle Aeronautical University",
                "University of Alabama--Huntsville",
                "University of Kentucky",
                "University of Miami",
                "University of Oklahoma",
                "University of South Carolina",
                "Kansas State University",
                "Oklahoma State University",
                "University of Kansas",
                "University of Maryland--Baltimore County",
                "University of Missouri",
                "Binghamton University--SUNY (Watson)",
                "Louisiana State University--Baton Rouge",
                "Southern Methodist University (Lyle)",
                "University of Arkansas--Fayetteville",
                "University of Georgia",
                "Utah State University",
                "Oregon Health and Science University",
                "University of California--Merced",
                "Clarkson University",
                "Marquette University",
                "Temple University",
                "Tulane University",
                "Virginia Commonwealth University",
                "West Virginia University (Statler)",
                "Baylor University",
                "CUNY--City College (Grove)",
                "Florida International University",
                "University of Louisville (Speed)",
                "University of Alabama",
                "University of Massachusetts--Lowell (Francis)",
                "University of Nevada--Reno",
                "Indiana University-Purdue University--Indianapolis",
                "Montana State University",
                "Santa Clara University",
                "Wayne State University",
                "Boise State University",
                "Ohio University (Russ)",
                "San Diego State University",
                "University of Alabama--Birmingham",
                "University of Colorado--Denver",
                "University of Idaho",
                "University of New Hampshire",
                "University of Tulsa",
                "University of Vermont",
                "University of Wisconsin--Milwaukee",
                "University of Wyoming",
                "Howard University",
                "New Mexico State University"
                )}


@app.route('/', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        return(render_template('index.html'))

    if flask.request.method == 'POST':
        gre = flask.request.form['gre']
        toefl = flask.request.form['toefl']
        ur = flask.request.form['university_rating']
        sop = flask.request.form['sop']
        lor = flask.request.form['lor']
        cgpa = flask.request.form['cgpa']
        rw = flask.request.form['research_work']
        input_variables = pd.DataFrame([[gre, toefl, ur, sop, lor, cgpa, rw]],
                                       columns=[
                                           'gre', 'toefl', 'university_rating', 'sop', 'lor', 'cgpa', 'research_work'],
                                       dtype=float)
        prediction = model.predict(input_variables)[0]
        return render_template('result.html',
                               original_input={'gre': gre,
                                               'toefl': toefl,
                                               'university_rating': ur,
                                               'sop': sop,
                                               'lor': lor,
                                               'cgpa': cgpa,
                                               'research_work': rw},
                               result=round(prediction[0]*100, 2),
                               Clg_List=Clg_List[int(ur)],
                               )


if __name__ == '__main__':
    app.run(debug=False, port=7000)