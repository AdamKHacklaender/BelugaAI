from __future__ import unicode_literals, print_function
import random
import warnings
from pathlib import Path
import spacy
from spacy.util import minibatch, compounding
from spacy import displacy


nlp = spacy.blank('en')


# data to train from
TRAIN_DATA = [('"They are (1) that the appellant\'s legal advisors failed to advise her correctly on matters that went to the heart of her plea because (2) the elements of the offence were not made out and the appellant could not have been convicted with the result that (3) the appellant\'s plea was equivocal.", \' If, on any version of the facts, the elements of the offence were not made out, this conviction would undeniably be unsafe and would fall to be quashed even though the appellant had pleaded guilty: see R v McReady and Hurd [1978] 1 WLR 1376', {'entities': [(500, 538, 'Case')]}), (' Section 2 of the Sexual Offences Act 2003 ("the Act") provides:\\\', \\\'"(1) A person (A) commits an offence if –\\\', \\\'(a) he intentionally penetrates the vagina … of another person (B) with a part of his body or anything else,\\\', \\\'(b) the penetration is sexual,\\\', \\\'(c) (B) does not consent to the penetration, and\\\', \\\'(d) (A) does not reasonably believe that (B) consents.\', " \'(2) Whether a belief is reasonable is to be determined having regard to all the circumstances, including any steps A has taken to ascertain whether B consents.", " \'This last provision has been considered in a number of decisions (in particular, R v Jheeta [2007] 2 Cr App R 34, R v Devonald [2008] EWCA Crim 527 and R v B [2013] EWCA Crim 823).", \' In R v EB [2006] EWCA Crim 2945, [2007] 1 WLR 1567, this court had to consider whether failure to disclose HIV status could vitiate consent (and, equally, belief in consent) to sexual intercourse', {'entities': [(1, 42, 'Statute'), (731, 762, 'Case'), (627, 658, 'Case'), (660, 693, 'Case'), (698, 724, 'Case'), (764, 781, 'Case')]}), (' In Assange v Swedish Prosecution Authority [2011] EWHC 2849 (Admin), the Divisional Court was concerned with the potential criminality in this country of having sexual intercourse without a condom when it had been made clear that consent was only forthcoming if a condom was used.\', " \'More recently, R(F) v DPP [2013] EWHC 945 (Admin) was concerned with a decision not to prosecute where the allegation was that consent was forthcoming on the basis that ejaculation would only take place outside the body.", \' v Emmett [1988] AC 773 at 782 or that they were vitiated having been induced by a fundamental mistake as to law or fact', {'entities': [(4, 60, 'Case'), (74, 90, 'Court'), (302, 328, 'Case'), (514, 541, 'Case')]}), (' The only basis on which Mr Wainwright can argue that this court should now intervene is that the appellant was wrongly advised and did not appreciate the elements of the offence to which she was pleading guilty (see Revitt, Borg and Barnes v DPP [2006] EWHC 2266 Admin)', {'entities': [(217, 269, 'Case')]}), (' None of this demonstrates that the case would have failed on evidential grounds and neither it, nor the dispute between the appellant, her parents and Mr Thomas start to justify the conclusion that "the defence would quite probably have succeeded" so that "a clear injustice has been done": see R v Boal [1992] QB 591 at 599H-600A.\', " \'In R v ZBT [2012] EWCA Crim 1727, this court rejected the proposition that step-siblings in a familial context were in a relationship which gave rise to trust and postulated, by way of example, that there was no duty of care owed by the appellant to his step sister."]', {'entities': [(341, 370, 'Case'), (296, 318, 'Case')]})]


nlp = nlp.to_disk("/Users/adamk.hacklander/PycharmProjects/Beluga/main_model")

@plac.annotations(
    model=("Model name. Defaults to blank 'en' model.", "option", "m", str),
    output_dir=("Optional output directory", "option", "o", Path),
    n_iter=("Number of training iterations", "option", "n", int),
)
def main(model=None, output_dir=nlp, n_iter=500):
    """Load the model, set up the pipeline and train the entity recognizer."""
    if model is not None:
        nlp = spacy.load(model)  # load existing spaCy model
        print("Loaded model '%s'" % model)
    else:
        nlp = spacy.blank("en")  # create blank Language class
        print("Created blank 'en' model")

    # create the built-in pipeline components and add them to the pipeline
    # nlp.create_pipe works for built-ins that are registered with spaCy
    if "ner" not in nlp.pipe_names:
        ner = nlp.create_pipe("ner")
        nlp.add_pipe(ner, last=True)
    # otherwise, get it so we can add labels
    else:
        ner = nlp.get_pipe("ner")

    # add labels
    for _, annotations in TRAIN_DATA:
        for ent in annotations.get("entities"):
            ner.add_label(ent[2])

    # get names of other pipes to disable them during training
    pipe_exceptions = ["ner", "trf_wordpiecer", "trf_tok2vec"]
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]
    # only train NER
    with nlp.disable_pipes(*other_pipes) and warnings.catch_warnings():
        # show warnings for misaligned entity spans once
        warnings.filterwarnings("once", category=UserWarning, module='spacy')

        # reset and initialize the weights randomly – but only if we're
        # training a new model
        if model is None:
            nlp.begin_training()
        for itn in range(n_iter):
            random.shuffle(TRAIN_DATA)
            losses = {}
            # batch up the examples using spaCy's minibatch
            batches = minibatch(TRAIN_DATA, size=compounding(4.0, 32.0, 1.001))
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(
                    texts,  # batch of texts
                    annotations,  # batch of annotations
                    drop=0.1,  # dropout - make it harder to memorise data
                    losses=losses,
                )
            print("Losses", losses)

    # test the trained model
    for text, _ in TRAIN_DATA:
        doc = nlp(text)
        print("Entities", [(ent.text, ent.label_) for ent in doc.ents])
        print("Tokens", [(t.text, t.ent_type_, t.ent_iob) for t in doc])

        # test the trained model
    with open('test_text.txt', 'r') as file:
        testec = file.read()
    doc = nlp(testec)
    print("Entities in '%s'" % testec)
    pandolin = displacy.serve(doc, style="ent")
    colors = {"ORG": "linear-gradient(90deg, #aa9cfc, #fc9ce7)"}
    options = {"ents": ["ORG"], "colors": colors}
    displacy.serve(doc, style="ent", options=options)
    for ent in doc.ents:
        print(ent.label_, ent.text)
        print(pandolin)


    # save model to output directory
    if output_dir is not None:
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir()
        nlp.to_disk(output_dir)
        print("Saved model to", output_dir)

        # test the saved model
        print("Loading from", output_dir)
        nlp2 = spacy.load(output_dir)
        for text, _ in TRAIN_DATA:
            doc = nlp2(text)
            print("Entities", [(ent.text, ent.label_) for ent in doc.ents])
            print("Tokens", [(t.text, t.ent_type_, t.ent_iob) for t in doc])


if __name__ == "__main__":
    plac.call(main)