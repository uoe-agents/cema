<<<<<<< HEAD:xavi/language.py
import logging
from typing import Tuple, List, Union

import numpy as np
import pandas as pd
import simplenlg as nlg
import re

from xavi.query import Query, QueryType
from xavi.matching import ActionGroup

logger = logging.getLogger(__name__)


class Language:
    REWARD_DICT = {
        "time": "our time to reach the goal",
        "coll": "a collision",
        "angular_velocity": "our lateral acceleration",
        "curvature": "our curvature",
        "jerk": "our jerk",
        "dead": "our goal"
    }
    VERBS_DICT = {
        "samevelocity": ("have", "the same speed as us"),
        "exitright": ("turn", "right"),
        "exitleft": ("turn", "left"),
        "exitstraight": ("go", "straight"),
        "changelaneleft": ("change", "lane to the left"),
        "changelaneright": ("change", "lane to the right"),
        "continue": ("go", "straight"),
        "faster": ("be", "faster than us"),
        "slower": ("be", "slower than us"),
        "decelerate": ("slow down", None),
        "accelerate": ("speed up", None),
        "maintain": ("maintain", "velocity"),
        "stops": ("stop", None)
    }
    FEATURE_SPLIT = re.compile(r"^(\w+)\(([^,]*)(,[^,]+)*\)$")
    CAMEL_SPLIT = re.compile(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)")

    def __init__(self,
                 n_associative: int = 2,
                 n_final: int = 2,
                 n_efficient: Tuple[int, int] = (1, 4)):
        """ Initialise a new explanation generation language class.
        This class uses SimpleNLG by Gatt and Reiter, 2009 to generate explanations.

        Args:
            n_associative: The number of associative causes to use for explanations.
            n_final: The number of final causes to use for explanations.
            n_efficient: The number of efficient causes (per cause time) to use for explanations.
        """
        self.n_associative = n_associative
        self.n_final = n_final
        self.n_efficient = n_efficient
        self.query = None

        self.__lexicon = nlg.Lexicon().getDefaultLexicon()
        self.__factory = nlg.NLGFactory(self.__lexicon)
        self.__realiser = nlg.Realiser(self.__lexicon)
        self.__tense_dict = {"past": nlg.Tense.PAST, "present": nlg.Tense.PRESENT, "future": nlg.Tense.FUTURE}

    def convert_to_sentence(self,
                            query: Query,
                            final_causes: pd.DataFrame = None,
                            efficient_causes: Tuple[pd.DataFrame, ...] = None,
                            action_group: ActionGroup = None) -> (str, str, str):
        """ Convert a set of causal attributions to a natural language reply.

        Args:
            query: The query being answered.
            final_causes: Causal attributions for the final explanation.
            efficient_causes: Causal attributions for the efficient explanations.
            action_group: The action to be executed by the vehicle.

        Returns:
            A tuple of explanations for each mode of reasoning.
        """
        self.query = query
        final_explanation = ""
        efficient_explanation = ""
        associative_explanation = ""

        if self.query.type == QueryType.WHAT:
            associative_explanation = self.__associative_explanation(action_group)
        elif self.query.type == QueryType.WHY:
            final_explanation = self.__teleological_explanation(final_causes, False)
            efficient_explanation = self.__mechanistic_explanation(efficient_causes, query)
        elif self.query.type == QueryType.WHY_NOT:
            final_explanation = self.__teleological_explanation(final_causes, True)
            efficient_explanation = self.__mechanistic_explanation(efficient_causes, query)
        elif self.query.type == QueryType.WHAT_IF:
            final_explanation = self.__teleological_explanation(final_causes, True)
            efficient_explanation = self.__mechanistic_explanation(efficient_causes, query)
            associative_explanation = self.__associative_explanation(action_group)

        return final_explanation, efficient_explanation, associative_explanation

    def __action_to_verb(self, actions: Union[str, List[str]], gen_str: bool = False) -> Union[nlg.VPPhraseSpec, str]:
        """ Convert an action from matching.py to a verb phrase. """
        def convert_action(action):
            camel_split = re.compile(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)")
            man_split = [man.lower() for man in camel_split.findall(action)]
            man = " ".join(man_split[0:2])
            man_verb = self.__factory.createVerbPhrase(man_split[0])
            if len(man_split) == 2:
                man_verb.addComplement(man_split[1])
            if len(man_split) > 2:
                if man == "change lane":
                    man_verb.addComplement(f"to the {man_split[2].lower()}")
                elif man == "go straight":
                    man_verb.addComplement(f"through the {man_split[2].lower()}")
            if gen_str:
                man_verb.setFeature(nlg.Feature.FORM, nlg.Form.GERUND)
                return self.__realiser.realise(man_verb).getRealisation()
            return man_verb
        if isinstance(actions, str):
            return convert_action(actions)
        elif isinstance(actions, list):
            return " and ".join([convert_action(a) for a in actions])

    def __feature_to_verb(self, feature: str) -> (int, nlg.VPPhraseSpec):
        splits = feature.split("_")
        vid = int(splits[0])
        action_split = splits[1:]
        if "macro" in action_split:
            action_split.remove("macro")
            match = self.FEATURE_SPLIT.match(action_split[0])
            action, params = match.groups()[0].lower(), match.groups()[1:]
            if "stop" in action.lower():
                action = "stops"
            else:
                params = " ".join([p for p in params if p is not None and "[" not in p])
                action = action + params
        else:
            action = ''.join(action_split).lower()

        if action in self.VERBS_DICT:
            verb, compl = self.VERBS_DICT[action]
            verb = self.__factory.createVerbPhrase(verb)
            if compl:
                verb.addComplement(compl)
        else:
            raise ValueError(f"Unknown action key: {action}")
        return vid, verb

    def __associative_explanation(self, action: ActionGroup, max_depth: int = 4) -> str:
        if action is None:
            return ""

        d = 0
        sentences = []
        if isinstance(action, list):
            action = action[0]
        for i, segment in enumerate(action.segments):
            clause = self.__factory.createCoordinatedPhrase()
            clause.setFeature(nlg.Feature.AGGREGATE_AUXILIARY, True)
            for action in segment.actions:
                segment_phrase = self.__action_to_verb(action)
                clause.addCoordinate(segment_phrase)
                print(self.__realiser.realise(clause).getRealisation())
                d += 1
                if d == max_depth: break
            tense = self.query.tense
            if self.query.t_action is not None:
                if self.query.t_query >= self.query.t_action:
                    tense = "past"
                elif self.query.t_query - self.query.t_action > -20:  # TODO: FPS is hardcoded
                    tense = "present"
                else:
                    tense = "future"
            clause.setTense(tense)
            if self.query.type == QueryType.WHAT_IF:
                clause.setFeature(nlg.Feature.MODAL, "would")
                if tense == "past":
                    clause.setFeature(nlg.Feature.PERFECT, True)
            if i > 0:
                clause.addPreModifier("then")
            sentence = self.__factory.createClause("we", clause)
            sentences.append(self.__realiser.realiseSentence(sentence))
            if d == max_depth: break
        return " ".join(sentences)

    def __teleological_explanation(self, causes: pd.DataFrame, contrastive: bool = False) -> str:
        def get_reward_verb(t, r, dr):
            if t == "dead":
                verb = self.__factory.createVerbPhrase("reach")
                verb.setNegated(bool(dr >= 0 and r == 1))
                if r == 0 or r == 1 and causes.loc[:, "reference"].sum() == 1:
                    verb.setPreModifier("always")
                elif r == 1 and causes.loc[:, "reference"].sum() != 1:
                    verb.addModifier("sometimes")
                if dr == 0:
                    verb.addModifier("still")
            elif t == "coll":
                if change < 0:
                    verb = self.__factory.createVerbPhrase("avoid")
                else:
                    verb = self.__factory.createVerbPhrase("cause")
                    verb.setNegated(bool(r < 1))
                    if dr == 0:
                        verb.addModifier("still")
            else:
                if dr == 0:
                    verb = self.__factory.createVerbPhrase("change")
                    verb.setNegated(True)
                else:
                    verb = self.__factory.createVerbPhrase("increase" if dr > 0.0 else "decrease")
            return verb

        causes.loc[causes.index != "time", :] *= -1
        sentences = []
        for i, reward_type in enumerate(causes.index):
            change = causes.loc[reward_type, "absolute"]
            ref = causes.loc[reward_type, "reference"]
            alt = causes.loc[reward_type, "alternative"]
            if contrastive:
                s_ref = self.__action_to_verb(self.query.action, True)
                if self.query.factual is not None:
                    s_alt = self.__action_to_verb(self.query.factual, True)
                else:
                    s_alt = s_ref
                    s_ref = "not " + s_ref
                if self.query.agent_id != 0:
                    s_ref = f"vehicle {self.query.agent_id} {s_ref}"
                    s_alt = f"vehicle {self.query.agent_id} {s_alt}"
                object = self.__factory.createNounPhrase(Language.REWARD_DICT.get(reward_type, "a change"))
                v_ref = get_reward_verb(reward_type, ref, change)
                v_alt = get_reward_verb(reward_type, alt, -change)

                c_ref = self.__factory.createClause(s_ref, v_ref, object)
                c_ref.setTense(self.__tense_dict[self.query.tense])
                c_ref.setFeature(nlg.Feature.MODAL, "would")
                c_alt = self.__factory.createClause(s_alt, v_alt, object)
                c_alt.setTense(self.__tense_dict[self.query.tense])
                sentences.append(self.__realiser.realiseSentence(c_ref))
                sentences.append(self.__realiser.realiseSentence(c_alt))
            else:
                subject = "it" if i != 0 else self.__action_to_verb(self.query.action, True)
                object = self.__factory.createNounPhrase(Language.REWARD_DICT.get(reward_type, "a change"))
                verb = get_reward_verb(reward_type, ref, change)

                clause = self.__factory.createClause(subject, verb, object)
                clause.setTense(self.__tense_dict[self.query.tense])
                sentences.append(self.__realiser.realiseSentence(clause))
        return " ".join(sentences)

    def __mechanistic_explanation(self, efficient_causes: Tuple[pd.DataFrame, ...], query: Query) -> str:
        def explain_causes(causes: pd.DataFrame, time: str):
            if causes is None:
                return ""
            causes = -(-causes.mean(0)).sort_values()
            causes = causes[causes > 0.0]
            sentences = []
            for i, feature_type in enumerate(causes.index):
                vid, verb = self.__feature_to_verb(feature_type)
                if time == "past":
                    verb.setTense(nlg.Tense.PAST)
                    if self.query.tense == "future":
                        verb.setTense(nlg.Tense.FUTURE)
                        if i == 0:
                            verb.addModifier("likely")
                        verb.setFeature(nlg.Feature.PERFECT, True)
                elif time == "future":
                    verb.setTense(nlg.Tense.FUTURE)
                    if i == 0:
                        verb.addModifier("likely")
                    if self.query.tense == "past":
                        verb.setTense(nlg.Tense.PRESENT)
                        verb.setFeature(nlg.Feature.MODAL, "would")
                        verb.setFeature(nlg.Feature.PERFECT, True)
                subject = "we" if vid == 0 else f"vehicle {vid}"
                sentence = self.__factory.createClause(subject, verb)
                if i == 0:
                    sentence.setFeature(nlg.Feature.CUE_PHRASE, "because")
                sentences.append(self.__realiser.realiseSentence(sentence))
            return " ".join(sentences)
        ep = explain_causes(efficient_causes[0], "past" if query.tense in ["past", "present"] else "present")
        ef = explain_causes(efficient_causes[1], "present" if query.tense in ["past", "present"] else "future")
        return "Past causes: " + ep + "; Future causes: " + ef
=======
import logging
from typing import Tuple, List, Union

import numpy as np
import pandas as pd
import simplenlg as nlg
import re

from .query import Query, QueryType
from .matching import ActionGroup

logger = logging.getLogger(__name__)


class Language:
    REWARD_DICT = {
        "time": "our time to reach the goal",
        "coll": "a collision",
        "angular_velocity": "our lateral acceleration",
        "curvature": "our curvature",
        "jerk": "our jerk",
        "dead": "our goal"
    }
    VERBS_DICT = {
        "samevelocity": ("have", "the same speed as us"),
        "exitright": ("turn", "right"),
        "exitleft": ("turn", "left"),
        "exitstraight": ("go", "straight"),
        "changelaneleft": ("change", "lane to the left"),
        "changelaneright": ("change", "lane to the right"),
        "continue": ("go", "straight"),
        "faster": ("be", "faster than us"),
        "slower": ("be", "slower than us"),
        "decelerate": ("slow down", None),
        "accelerate": ("speed up", None),
        "maintain": ("maintain", "velocity"),
        "stops": ("stop", None)
    }
    FEATURE_SPLIT = re.compile(r"^(\w+)\(([^,]*)(,[^,]+)*\)$")
    CAMEL_SPLIT = re.compile(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)")

    def __init__(self,
                 n_associative: int = 2,
                 n_final: int = 2,
                 n_efficient: Tuple[int, int] = (1, 4)):
        """ Initialise a new explanation generation language class.
        This class uses SimpleNLG by Gatt and Reiter, 2009 to generate explanations.

        Args:
            n_associative: The number of associative causes to use for explanations.
            n_final: The number of final causes to use for explanations.
            n_efficient: The number of efficient causes (per cause time) to use for explanations.
        """
        self.n_associative = n_associative
        self.n_final = n_final
        self.n_efficient = n_efficient
        self.query = None

        self.__lexicon = nlg.Lexicon().getDefaultLexicon()
        self.__factory = nlg.NLGFactory(self.__lexicon)
        self.__realiser = nlg.Realiser(self.__lexicon)
        self.__tense_dict = {"past": nlg.Tense.PAST, "present": nlg.Tense.PRESENT, "future": nlg.Tense.FUTURE}

    def convert_to_sentence(self,
                            query: Query,
                            final_causes: pd.DataFrame = None,
                            efficient_causes: Tuple[pd.DataFrame, ...] = None,
                            action_group: ActionGroup = None) -> (str, str, str):
        """ Convert a set of causal attributions to a natural language reply.

        Args:
            query: The query being answered.
            final_causes: Causal attributions for the final explanation.
            efficient_causes: Causal attributions for the efficient explanations.
            action_group: The action to be executed by the vehicle.

        Returns:
            A tuple of explanations for each mode of reasoning.
        """
        self.query = query
        final_explanation = ""
        efficient_explanation = ""
        associative_explanation = ""

        if self.query.type == QueryType.WHAT:
            associative_explanation = self.__associative_explanation(action_group)
        elif self.query.type == QueryType.WHY:
            final_explanation = self.__teleological_explanation(final_causes, False)
            efficient_explanation = self.__mechanistic_explanation(efficient_causes, query)
        elif self.query.type == QueryType.WHY_NOT:
            final_explanation = self.__teleological_explanation(final_causes, True)
            efficient_explanation = self.__mechanistic_explanation(efficient_causes, query)
        elif self.query.type == QueryType.WHAT_IF:
            final_explanation = self.__teleological_explanation(final_causes, True)
            efficient_explanation = self.__mechanistic_explanation(efficient_causes, query)
            associative_explanation = self.__associative_explanation(action_group)

        return final_explanation, efficient_explanation, associative_explanation

    def __action_to_verb(self, actions: Union[str, List[str]], gen_str: bool = False) -> Union[nlg.VPPhraseSpec, str]:
        """ Convert an action from matching.py to a verb phrase. """
        def convert_action(action):
            camel_split = re.compile(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)")
            man_split = [man.lower() for man in camel_split.findall(action)]
            man = " ".join(man_split[0:2])
            man_verb = self.__factory.createVerbPhrase(man_split[0])
            if len(man_split) == 2:
                man_verb.addComplement(man_split[1])
            if len(man_split) > 2:
                if man == "change lane":
                    man_verb.addComplement(f"to the {man_split[2].lower()}")
                elif man == "go straight":
                    man_verb.addComplement(f"through the {man_split[2].lower()}")
            if gen_str:
                man_verb.setFeature(nlg.Feature.FORM, nlg.Form.GERUND)
                return self.__realiser.realise(man_verb).getRealisation()
            return man_verb
        if isinstance(actions, str):
            return convert_action(actions)
        elif isinstance(actions, list):
            return " and ".join([convert_action(a) for a in actions])

    def __feature_to_verb(self, feature: str) -> (int, nlg.VPPhraseSpec):
        splits = feature.split("_")
        vid = int(splits[0])
        action_split = splits[1:]
        if "macro" in action_split:
            action_split.remove("macro")
            match = self.FEATURE_SPLIT.match(action_split[0])
            action, params = match.groups()[0].lower(), match.groups()[1:]
            if "stop" in action.lower():
                action = "stops"
            else:
                params = " ".join([p for p in params if p is not None and "[" not in p])
                action = action + params
        else:
            action = ''.join(action_split).lower()

        if action in self.VERBS_DICT:
            verb, compl = self.VERBS_DICT[action]
            verb = self.__factory.createVerbPhrase(verb)
            if compl:
                verb.addComplement(compl)
        else:
            raise ValueError(f"Unknown action key: {action}")
        return vid, verb

    def __associative_explanation(self, action: ActionGroup, max_depth: int = 4) -> str:
        if action is None:
            return ""

        d = 0
        sentences = []
        if isinstance(action, list):
            action = action[0]
        for i, segment in enumerate(action.segments):
            clause = self.__factory.createCoordinatedPhrase()
            clause.setFeature(nlg.Feature.AGGREGATE_AUXILIARY, True)
            for action in segment.actions:
                segment_phrase = self.__action_to_verb(action)
                clause.addCoordinate(segment_phrase)
                print(self.__realiser.realise(clause).getRealisation())
                d += 1
                if d == max_depth: break
            tense = self.query.tense
            if self.query.t_action is not None:
                if self.query.t_query >= self.query.t_action:
                    tense = "past"
                elif self.query.t_query - self.query.t_action > -20:  # TODO: FPS is hardcoded
                    tense = "present"
                else:
                    tense = "future"
            clause.setTense(tense)
            if self.query.type == QueryType.WHAT_IF:
                clause.setFeature(nlg.Feature.MODAL, "would")
                if tense == "past":
                    clause.setFeature(nlg.Feature.PERFECT, True)
            if i > 0:
                clause.addPreModifier("then")
            sentence = self.__factory.createClause("we", clause)
            sentences.append(self.__realiser.realiseSentence(sentence))
            if d == max_depth: break
        return " ".join(sentences)

    def __teleological_explanation(self, causes: pd.DataFrame, contrastive: bool = False) -> str:
        def get_reward_verb(t, r, dr):
            if t == "dead":
                verb = self.__factory.createVerbPhrase("reach")
                verb.setNegated(bool(dr >= 0 and r == 1))
                if r == 0 or r == 1 and causes.loc[:, "reference"].sum() == 1:
                    verb.setPreModifier("always")
                elif r == 1 and causes.loc[:, "reference"].sum() != 1:
                    verb.addModifier("sometimes")
                if dr == 0:
                    verb.addModifier("still")
            elif t == "coll":
                if change < 0:
                    verb = self.__factory.createVerbPhrase("avoid")
                else:
                    verb = self.__factory.createVerbPhrase("cause")
                    verb.setNegated(bool(r < 1))
                    if dr == 0:
                        verb.addModifier("still")
            else:
                if dr == 0:
                    verb = self.__factory.createVerbPhrase("change")
                    verb.setNegated(True)
                else:
                    verb = self.__factory.createVerbPhrase("increase" if dr > 0.0 else "decrease")
            return verb

        causes.loc[causes.index != "time", :] *= -1
        sentences = []
        for i, reward_type in enumerate(causes.index):
            change = causes.loc[reward_type, "absolute"]
            ref = causes.loc[reward_type, "reference"]
            alt = causes.loc[reward_type, "alternative"]
            if contrastive:
                s_ref = self.__action_to_verb(self.query.action, True)
                if self.query.factual is not None:
                    s_alt = self.__action_to_verb(self.query.factual, True)
                else:
                    s_alt = s_ref
                    s_ref = "not " + s_ref
                if self.query.agent_id != 0:
                    s_ref = f"vehicle {self.query.agent_id} {s_ref}"
                    s_alt = f"vehicle {self.query.agent_id} {s_alt}"
                object = self.__factory.createNounPhrase(Language.REWARD_DICT.get(reward_type, "a change"))
                v_ref = get_reward_verb(reward_type, ref, change)
                v_alt = get_reward_verb(reward_type, alt, -change)

                c_ref = self.__factory.createClause(s_ref, v_ref, object)
                c_ref.setTense(self.__tense_dict[self.query.tense])
                c_ref.setFeature(nlg.Feature.MODAL, "would")
                c_alt = self.__factory.createClause(s_alt, v_alt, object)
                c_alt.setTense(self.__tense_dict[self.query.tense])
                sentences.append(self.__realiser.realiseSentence(c_ref))
                sentences.append(self.__realiser.realiseSentence(c_alt))
            else:
                subject = "it" if i != 0 else self.__action_to_verb(self.query.action, True)
                object = self.__factory.createNounPhrase(Language.REWARD_DICT.get(reward_type, "a change"))
                verb = get_reward_verb(reward_type, ref, change)

                clause = self.__factory.createClause(subject, verb, object)
                clause.setTense(self.__tense_dict[self.query.tense])
                sentences.append(self.__realiser.realiseSentence(clause))
        return " ".join(sentences)

    def __mechanistic_explanation(self, efficient_causes: Tuple[pd.DataFrame, ...], query: Query) -> str:
        def explain_causes(causes: pd.DataFrame, time: str):
            if causes is None:
                return ""
            causes = -(-causes.mean(0)).sort_values()
            causes = causes[causes > 0.0]
            sentences = []
            for i, feature_type in enumerate(causes.index):
                vid, verb = self.__feature_to_verb(feature_type)
                if time == "past":
                    verb.setTense(nlg.Tense.PAST)
                    if self.query.tense == "future":
                        verb.setTense(nlg.Tense.FUTURE)
                        if i == 0:
                            verb.addModifier("likely")
                        verb.setFeature(nlg.Feature.PERFECT, True)
                elif time == "future":
                    verb.setTense(nlg.Tense.FUTURE)
                    if i == 0:
                        verb.addModifier("likely")
                    if self.query.tense == "past":
                        verb.setTense(nlg.Tense.PRESENT)
                        verb.setFeature(nlg.Feature.MODAL, "would")
                        verb.setFeature(nlg.Feature.PERFECT, True)
                subject = "we" if vid == 0 else f"vehicle {vid}"
                sentence = self.__factory.createClause(subject, verb)
                if i == 0:
                    sentence.setFeature(nlg.Feature.CUE_PHRASE, "because")
                sentences.append(self.__realiser.realiseSentence(sentence))
            return " ".join(sentences)
        ep = explain_causes(efficient_causes[0], "past" if query.tense in ["past", "present"] else "present")
        ef = explain_causes(efficient_causes[1], "present" if query.tense in ["past", "present"] else "future")
        return "Past causes: " + ep + "; Future causes: " + ef
>>>>>>> 90fede0243d6f17b426f2d2d9ba80de72684e61a:src/cema/xavi/language.py
