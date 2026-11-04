"""Read-only source comparison against the official PyPI release; no installs/imports."""
import datetime,hashlib,io,json,urllib.request,urllib.parse,zipfile
from pathlib import Path
ROOT=Path('/ml2/alz')
SITE=Path('/ml2/codex-storage/future-workspaces/2026-10-27/task-2/benchmark/current_base_20261029/runtime/lib/python3.12/site-packages')
sha=lambda b:hashlib.sha256(b).hexdigest()
def main():
 url='https://pypi.org/pypi/particlegan/json'
 with urllib.request.urlopen(url,timeout=15) as response:raw=response.read(8*1024*1024+1)
 assert len(raw)<=8*1024*1024;data=json.loads(raw);version=data['info']['version']
 wheels=[q for q in data['urls'] if q['filename']==f'particlegan-{version}-py3-none-any.whl'];assert len(wheels)==1;w=wheels[0]
 parsed=urllib.parse.urlparse(w['url']);assert parsed.scheme=='https' and parsed.hostname=='files.pythonhosted.org'
 with urllib.request.urlopen(w['url'],timeout=15) as response:payload=response.read(16*1024*1024+1)
 assert len(payload)<=16*1024*1024 and sha(payload)==w['digests']['sha256']
 metadata=list(SITE.glob('particlegan-*.dist-info/METADATA'));assert len(metadata)==1
 installed_version=next(q.split(': ',1)[1] for q in metadata[0].read_text().splitlines() if q.startswith('Version: '));assert installed_version==version,'Latest official release differs; do not silently upgrade/restart scientific jobs'
 differences=[];inventory={}
 with zipfile.ZipFile(io.BytesIO(payload)) as archive:
  files=sorted(n for n in archive.namelist() if n.startswith('particlegan/') and n.endswith('.py'))
  installed=sorted(str(p.relative_to(SITE)) for p in (SITE/'particlegan').rglob('*.py'));assert installed==files,'Installed Python source cohort differs from official wheel'
  for name in files:
   p=SITE/name;b=p.read_bytes();expected=archive.read(name);inventory[str(p)]=sha(b)
   if b!=expected:differences.append(name)
 assert not differences
 leader_path=ROOT/'analysis/pure_formulations_leaderboard.json';leader=json.loads(leader_path.read_text())
 result={'status':'VERIFIED_CURRENT_OFFICIAL_PARTICLEGAN_RELEASE_AND_EXACT_INSTALLED_SOURCE','checked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'installed_metadata_version':installed_version,'latest_official_pypi_version':version,'release_uploaded_utc':w['upload_time_iso_8601'],'wheel_SHA256':sha(payload),'official_wheel_python_files_compared':len(files),'changed_or_missing_installed_source_files':differences,'installed_source_SHA256':inventory,
 'API_used':['get_recipe','Recipe.make_prior','Recipe.make_optimizers','Recipe.make_critic_penalty','Recipe.make_loss','Recipe.make_prior_regularizer','UpdatePolicy','serial_autograd'],'training_loop':'ALZ caller-owned count-model loop using native recipe factories/UpdatePolicy; not the GANTrainer convenience loop','completed_pure_comparisons':len(leader['rows']),'completed_native_recipe_families':leader['native_recipe_families_tested'],'selected_pure_model':json.loads((ROOT/'pure_incumbent.json').read_text())['pure_formulation'],
 'coverage_limits':'Completed optimizedBISPF, originalBISPF andBCAP in ALZ adapter;13pure comparisons include retainedarchitecture/training/observation variations and one fixedpurewholebag ensemble. This does not assert that every named library model family or preset was trained. E22/MoG/DDGAN/AE-VAE families not claimed tested. No experimental newer branch substituted for published release.',
 'runtime_mutated':False,'CUDA_context_created':False,'sources':[url,'https://pypi.org/project/particlegan/',w['url']],'metadata_response_SHA256':sha(raw),'source_SHA256':{str(Path(__file__).resolve()):sha(Path(__file__).resolve().read_bytes()),str(leader_path):sha(leader_path.read_bytes())}}
 out=ROOT/'analysis/particlegan_release_verification_20261104.json';assert not out.exists();out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:result[k] for k in ['status','checked_utc','latest_official_pypi_version','installed_metadata_version','official_wheel_python_files_compared','completed_pure_comparisons','completed_native_recipe_families','runtime_mutated']},indent=2))
if __name__=='__main__':main()
