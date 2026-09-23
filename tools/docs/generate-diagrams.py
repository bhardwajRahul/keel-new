#!/usr/bin/env python3
"""Generate the documentation's standalone, accessible SVG diagrams (stdlib only)."""
from pathlib import Path
from html import escape

DEST = Path(__file__).resolve().parents[2] / 'docs'
INK = '#172234'
MUTED = '#576880'
BLUE = '#2563eb'
LINE = '#c8d4e4'
TINT = '#edf4ff'

class Figure:
    def __init__(self, key, title, description, height=900):
        self.key, self.height = key, height
        self.parts = [f'''<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="{height}" viewBox="0 0 1280 {height}" role="img" aria-labelledby="title desc">
<title id="title">{escape(title)}</title><desc id="desc">{escape(description)}</desc>
<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0L10 5L0 10Z" fill="{BLUE}"/></marker></defs>
<rect width="1280" height="{height}" fill="#ffffff"/>
<style>text{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Arial,sans-serif;fill:{INK}}}.mono{{font-family:ui-monospace,SFMono-Regular,Menlo,monospace}}.muted{{fill:{MUTED}}}</style>''']
        self.text(40,42,key,15,'mono muted')
        self.text(40,100,title,38)
    def text(self,x,y,content,size=19,cls='',anchor='start'):
        self.parts.append(f'<text x="{x}" y="{y}" font-size="{size}" class="{cls}" text-anchor="{anchor}">{escape(content)}</text>')
    def box(self,x,y,w,h,title,lines=(),tint=False):
        self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="7" fill="{TINT if tint else "#fff"}" stroke="{BLUE if tint else LINE}" stroke-width="1.5"/>')
        self.text(x+20,y+32,title,21)
        for i,line in enumerate(lines):self.text(x+20,y+62+i*26,line,17,'muted')
    def arrow(self,points,label=None,labelxy=None):
        self.parts.append(f'<polyline points="{points}" fill="none" stroke="{BLUE}" stroke-width="2" stroke-linejoin="round" marker-end="url(#arrow)"/>')
        if label:self.text(*labelxy,label,15,'mono muted')
    def footer(self,note):
        y=self.height-66
        self.parts.append(f'<path d="M40 {y}H1240" stroke="{LINE}"/>')
        self.text(40,y+32,note,15,'mono muted')
    def save(self,name):
        destination = DEST/name
        destination.parent.mkdir(parents=True,exist_ok=True)
        destination.write_text('\n'.join(self.parts)+ '\n</svg>\n')

f=Figure('01 / implemented architecture','the selector chooses. the host checks.',
          'Fresh unpinned tasks receive prepared route choices. One backend selects or abstains; the host rechecks. Embedded DeepSeek uses host-enforced tool focus. External ACP agents keep their own internal loops.',880)
for x,title,lines,tint in [
 (40,'fresh, unpinned task',['pinned and existing sessions','keep their current route'],False),
 (350,'prepare routes',['installed, enabled providers','and eligible model options'],False),
 (660,'selected backend',['local laya or hosted jev','returns an id or abstains'],True),
 (970,'host rechecks',['current state and eligibility','invalid choice → fallback'],False)]:
 f.box(x,160,270,125,title,lines,tint)
for x in (310,620,930):f.arrow(f'{x},222 {x+40},222')
f.arrow('1105,285 1105,330 640,330 640,365')
f.text(640,397,'coding adapter determines who owns the next loop',19,'', 'middle')
f.arrow('640,410 325,410 325,450')
f.arrow('640,410 955,410 955,450')
f.box(40,450,570,290,'embedded deepseek',[],True)
f.box(670,450,570,290,'external acp agent')
for i,t in enumerate(['1. selector chooses inspect / implement / verify / answer','2. host prepares and advertises allowed tools','3. coding model proposes output or tool calls','4. host checks the tool and permissions before dispatch']):
 f.text(60,525+i*47,t,17)
for i,t in enumerate(['1. host starts the selected provider session','2. provider owns its internal tool loop','3. compatible sessions can receive installed','   codex computer-use mcp tools']):
 f.text(690,525+i*47,t,17)
f.text(40,778,'normal mode skips selection; ordinary host permissions still apply.',18,'muted')
f.footer('implemented boundaries · selecting an action never grants permission')
f.save('diagrams/architecture.svg')

f=Figure('02 / decision modes','three modes. one host in control.',
          'Laya runs locally with its worker and model. Jev is opt-in and uses a protected credential for TypeSafe. Normal mode skips selector calls. A saved Laya preference can remain pending while installation finishes.',760)
rows=[(165,'laya / default',['local core ml worker','pinned model files required','no hosted selector call'],True),
      (335,'jev / opt-in',['direct typesafe request','existing protected credential','no key-entry screen'],False),
      (505,'normal / bypass',['skip the selector','ordinary route behavior','provider setup still required'],False)]
for y,title,lines,tint in rows:
 f.box(40,y,570,145,title,lines,tint)
 if y<500:
  f.arrow(f'610,{y+73} 800,{y+73} 800,323 900,323')
 else:f.arrow(f'610,{y+73} 900,{y+73}')
f.box(900,260,340,130,'host checks the choice',['validate eligible route','apply or use fallback'],True)
f.box(900,520,340,100,'ordinary harness',['existing permissions remain'])
f.text(40,132,'while laya installs: save the preference; keep the ordinary harness usable.',18,'muted')
f.footer('local describes the selector · coding workers may still use hosted models')
f.save('diagrams/decision-modes.svg')

f=Figure('03 / proposed improvement loop','keep the evidence. review the change.',
          'Proposed process, not automatic training: record a decision, create a replay case, compare baseline and candidate, retain baseline if worse, and require human approval before a versioned change.',830)
f.box(40,170,270,145,'decision record',['options and selected id','host check and fallback','observed result'])
f.box(350,170,270,145,'replay case',['relevant task state','expected behavior','repeatable conditions'])
f.box(660,170,270,145,'baseline + candidate',['same tasks and scoring','one intentional change','held-out final task set'],True)
f.box(970,170,270,145,'compare outcomes',['task quality and failures','total time and cost','review effort'])
for x in (310,620,930):f.arrow(f'{x},243 {x+40},243')
f.arrow('1105,315 1105,375 895,375 895,450')
f.box(730,450,330,120,'human review',['inspect gains and regressions','approve only with evidence'],True)
f.arrow('1060,510 1140,510 1140,635 950,635')
f.box(660,580,290,120,'versioned change',['keep a rollback target','use in future runs'])
f.arrow('730,510 565,510','no improvement',(570,493))
f.box(285,450,280,120,'keep baseline',['reject the candidate','keep the failure case'])
f.arrow('660,640 175,640 175,315','approved version → future records',(205,623))
f.text(40,746,'proposed workflow only; the current app records decisions and does not train itself.',18,'muted')
f.footer('evaluation design · no coding-quality improvement is claimed here')
f.save('proposals/improvement-loop.svg')
print('generated 3 documentation SVGs')
